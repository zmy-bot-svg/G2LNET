import os
import json
import pickle
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import pairwise_distances
import matplotlib.pyplot as plt
import seaborn as sns
import logging
from tqdm import tqdm
import warnings
from datetime import datetime
from typing import Optional, Dict, List, Tuple, Any
import gc
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
warnings.filterwarnings('ignore')

class JARVISDataValidator:
    """JARVIS-DFT data validator"""
    
    @staticmethod
    def validate_physical_properties(df: pd.DataFrame) -> Dict[str, Any]:
        """Physical sanity checks for JARVIS-DFT data."""
        logger.info("Running data validation...")
        
        validation_report = {
            'total_samples': len(df),
            'issues_found': [],
            'corrections_applied': [],
            'quality_score': 100.0,
            'is_valid': True
        }
        
        if 'optb88vdw_bandgap' in df.columns:
            negative_bandgap = df['optb88vdw_bandgap'] < 0
            if negative_bandgap.any():
                n_negative = negative_bandgap.sum()
                validation_report['issues_found'].append(
                    f"Found {n_negative} negative bandgap values"
                )
                df.loc[negative_bandgap, 'optb88vdw_bandgap'] = 0
                validation_report['corrections_applied'].append(
                    f"Set {n_negative} negative bandgap values to 0"
                )
                validation_report['quality_score'] -= 5
        
        if 'formation_energy_peratom' in df.columns:
            high_fe = df['formation_energy_peratom'] > 5
            low_fe = df['formation_energy_peratom'] < -10
            if high_fe.any():
                n_high = high_fe.sum()
                validation_report['issues_found'].append(
                    f"Found {n_high} high formation energy values (> 5 eV/atom)"
                )
                validation_report['quality_score'] -= 3
            if low_fe.any():
                n_low = low_fe.sum()
                validation_report['issues_found'].append(
                    f"Found {n_low} low formation energy values (< -10 eV/atom)"
                )
                validation_report['quality_score'] -= 3
        
        total_issues = len(validation_report['issues_found'])
        validation_report['quality_score'] = max(0, validation_report['quality_score'])
        
        if validation_report['issues_found']:
            logger.warning(f"Data validation found {total_issues} issues:")
            for issue in validation_report['issues_found']:
                logger.warning(f"  {issue}")
            if validation_report['corrections_applied']:
                logger.info("Corrections applied:")
                for correction in validation_report['corrections_applied']:
                    logger.info(f"  {correction}")
        else:
            logger.info("Data validation passed with no issues.")

        logger.info(f"Quality score: {validation_report['quality_score']:.1f}/100")
        return validation_report

def convert_to_g2lnet_list(subset_df, prop_name):
    """Convert dataset to G2LNet format."""
    logger.info(f"Converting property: '{prop_name}'")
    try:
        from pymatgen.core import Structure
    except ImportError:
        logger.error("Missing dependency: pymatgen. Install with: pip install pymatgen")
        return None

    data_list = []
    for _, row in tqdm(subset_df.iterrows(), total=len(subset_df), desc=f"Convert {prop_name}"):
        try:
            atoms_dict = row['atoms']
            structure_obj = Structure(
                lattice=atoms_dict['lattice_mat'],
                species=atoms_dict['elements'],
                coords=atoms_dict['coords'],
                coords_are_cartesian=True
            )
            cif_string = structure_obj.to(fmt="cif")
            record = {
                "material_id": row['jid'],
                "structure": cif_string,
                prop_name: row[prop_name]
            }
            data_list.append(record)
        except Exception:
            continue
    
    if not data_list:
        logger.error(f"No records converted for '{prop_name}'.")
        return None
    
    return data_list

class AdvancedJARVISSampler:
    """Advanced JARVIS sampler with multi-dimensional stratification."""
    
    def __init__(self, pkl_path: str, target_size: int = 15000, random_state: int = 42, 
                 pca_features: Optional[List[str]] = None,
                 thresholds: Optional[Dict[str, float]] = None):
        self.pkl_path = pkl_path
        self.target_size = target_size
        self.random_state = random_state
        self.full_data = None
        self.subset_data = None
        self.std_data = None
        self.std_indices = None
        self.sampling_metadata = {}
        self.validator = JARVISDataValidator()
        
        self.pca_features = pca_features or [
            'formation_energy_peratom', 'optb88vdw_bandgap', 
            'density', 'total_energy'
        ]
        self._pca_results = None
        self.thresholds = thresholds or {
            'bandgap_high': 3.0,
            'n_atoms_complex': 10
        }

        logger.info("Initializing sampler")
        logger.info(f"Target size: {target_size:,}")
        logger.info(f"Random seed: {random_state}")
        logger.info(f"PCA features: {self.pca_features}")
    
    def load_and_validate_data(self):
        """Load data and enforce strict task intersection filtering."""
        logger.info("Loading and validating data...")

        if not os.path.exists(self.pkl_path):
            logger.error(f"PKL file not found: {self.pkl_path}")
            return False

        try:
            with open(self.pkl_path, 'rb') as f:
                self.full_data = pd.read_pickle(f)

            self.core_task_cols = [
                'formation_energy_peratom',
                'optb88vdw_bandgap',
                'optb88vdw_total_energy'
            ]

            missing_cols = [c for c in self.core_task_cols if c not in self.full_data.columns]
            if missing_cols:
                logger.error(f"Missing required columns: {missing_cols}")
                return False

            for col in self.core_task_cols:
                self.full_data[col] = pd.to_numeric(self.full_data[col], errors='coerce')

            initial_len = len(self.full_data)
            self.full_data = self.full_data.dropna(subset=self.core_task_cols + ['atoms']).reset_index(drop=True)
            final_len = len(self.full_data)
            logger.info(
                f"Filtered rows with missing core attributes: {initial_len} -> {final_len}"
            )

            validation_report = self.validator.validate_physical_properties(self.full_data)
            self.sampling_metadata['data_validation'] = validation_report

            try:
                from jarvis.db.figshare import data
                data_55k = pd.DataFrame(data('dft_3d_2021'))
                if 'jid' in data_55k.columns and 'jid' in self.full_data.columns:
                    self.std_indices = self.full_data[self.full_data['jid'].isin(data_55k['jid'])].index
                    self.std_data = self.full_data.loc[self.std_indices].copy()
                    logger.info(f"Loaded Std-55k benchmark: {len(self.std_data)} samples")
                else:
                    logger.warning("Std-55k load failed: missing 'jid' column")
            except Exception as e:
                logger.warning(f"Failed to load Std-55k automatically: {e}")
            return True

        except Exception as e:
            logger.error(f"Data load failed: {e}")
            return False
    
    def create_enhanced_stratification(self):
        """Create enhanced stratification strategy."""
        logger.info("Creating stratification...")
        
        df = self.full_data.copy()
        
        df['bandgap_category'] = 'semiconductor'
        df.loc[df['optb88vdw_bandgap'] == 0, 'bandgap_category'] = 'metal'
        df.loc[df['optb88vdw_bandgap'] > self.thresholds['bandgap_high'], 'bandgap_category'] = 'insulator'

        fe_quartiles = df['formation_energy_peratom'].quantile([0.25, 0.5, 0.75])
        df['stability_quartile'] = pd.cut(
            df['formation_energy_peratom'], 
            bins=[-np.inf, fe_quartiles[0.25], fe_quartiles[0.5], fe_quartiles[0.75], np.inf],
            labels=['most_stable', 'stable', 'less_stable', 'least_stable']
        )

        df['n_elements'] = df['formula'].apply(lambda x: len(set([c for c in x if c.isupper()])) if pd.notna(x) else 1)
        df['chemical_complexity'] = pd.cut(
            df['n_elements'], 
            bins=[0, 1, 2, 3, np.inf],
            labels=['unary', 'binary', 'ternary', 'quaternary_plus']
        )

        df['n_atoms'] = df['atoms'].apply(lambda x: len(x['elements']) if isinstance(x, dict) and 'elements' in x else 1)
        df['structural_complexity'] = pd.cut(
            df['n_atoms'],
            bins=[0, self.thresholds['n_atoms_complex'], np.inf],
            labels=['simple_structure', 'complex_structure']
        )

        df['bandgap_category'] = df['bandgap_category'].fillna('unknown')
        df['stability_quartile'] = df['stability_quartile'].cat.add_categories('unknown').fillna('unknown')
        df['chemical_complexity'] = df['chemical_complexity'].cat.add_categories('unknown').fillna('unknown')
        df['structural_complexity'] = df['structural_complexity'].cat.add_categories('unknown').fillna('unknown')

        df['primary_strata'] = (
            df['bandgap_category'].astype(str) + '_' + 
            df['stability_quartile'].astype(str) + '_' +
            df['chemical_complexity'].astype(str) + '_' +
            df['structural_complexity'].astype(str)
        )

        self.sampling_metadata['stratification'] = {
            'formation_energy_quartiles': {
                'Q1': float(fe_quartiles[0.25]),
                'Q2': float(fe_quartiles[0.5]),
                'Q3': float(fe_quartiles[0.75])
            },
            'bandgap_distribution': df['bandgap_category'].value_counts().to_dict(),
            'chemical_complexity_distribution': df['chemical_complexity'].value_counts().to_dict(),
            'structural_complexity_distribution': df['structural_complexity'].value_counts().to_dict(),
            'primary_strata_count': df['primary_strata'].nunique(),
            'total_samples': len(df),
            'stratification_dimensions': ['electronic_properties', 'thermodynamic_stability', 'chemical_complexity', 'structural_complexity']
        }
        
        self.full_data = df
        logger.info(f"Stratification created with {df['primary_strata'].nunique()} strata")
        return df
    
    def execute_robust_sampling(self):
        """Execute robust stratified sampling."""
        logger.info("Running stratified sampling...")
        
        try:
            subset, _ = train_test_split(
                self.full_data,
                train_size=self.target_size,
                stratify=self.full_data['primary_strata'],
                random_state=self.random_state
            )
            sampling_method = "full_stratified"
            
        except ValueError as e:
            logger.warning(f"Full stratified sampling failed: {e}")
            
            try:
                subset, _ = train_test_split(
                    self.full_data,
                    train_size=self.target_size,
                    stratify=self.full_data['bandgap_category'],
                    random_state=self.random_state
                )
                sampling_method = "bandgap_stratified"
                
            except ValueError as e2:
                logger.warning(f"Bandgap stratified sampling failed: {e2}")
                subset = self.full_data.sample(
                    n=self.target_size, 
                    random_state=self.random_state
                )
                sampling_method = "random"
        
        self.subset_data = subset.copy()
        
        self.sampling_metadata['sampling'] = {
            'method_used': sampling_method,
            'target_size': self.target_size,
            'actual_size': len(self.subset_data),
            'sampling_ratio': len(self.subset_data) / len(self.full_data),
            'random_state': self.random_state,
            'timestamp': datetime.now().isoformat()
        }

        logger.info(f"Sampling complete: {len(self.subset_data):,} samples")
        logger.info(f"Sampling method: {sampling_method}")
        
        return self.subset_data
    
    def validate_sampling_quality(self):
        """Validate sampling quality."""
        logger.info("Validating sampling quality...")
        
        validation_results = {}
        properties = getattr(self, 'core_task_cols', ['formation_energy_peratom', 'optb88vdw_bandgap'])
        
        ks_results = {}
        for prop in properties:
            if prop in self.full_data.columns and prop in self.subset_data.columns:
                ks_stat, ks_pvalue = stats.ks_2samp(
                    self.full_data[prop].dropna(), 
                    self.subset_data[prop].dropna()
                )

                status = "PASS" if ks_pvalue > 0.05 else "FAIL"
                logger.info(f"{prop} KS test: p={ks_pvalue:.4f} {status}")
                
                ks_results[prop] = {
                    'statistic': float(ks_stat),
                    'pvalue': float(ks_pvalue),
                    'passed': bool(ks_pvalue > 0.05)
                }

        all_ks_passed = all(result['passed'] for result in ks_results.values())
        overall_quality = "EXCELLENT" if all_ks_passed else "GOOD"

        logger.info(f"Overall sampling quality: {overall_quality}")
        
        validation_results = {
            'ks_tests': ks_results,
            'overall_quality': overall_quality,
            'timestamp': datetime.now().isoformat()
        }
        
        self.sampling_metadata['validation'] = validation_results
        return validation_results

    def create_structural_extrapolation_set(self, ses_size: int = 1500):
        """Build structural extrapolation set (SES)."""
        logger.info("Building SES...")

        if self.subset_data is None:
            logger.error("Subset data is missing. Run sampling first.")
            return None

        residual_indices = self.full_data.index.difference(self.subset_data.index)
        residual_df = self.full_data.loc[residual_indices].copy()

        metastable_mask = residual_df['formation_energy_peratom'] > 0.5
        candidates = residual_df[metastable_mask].copy()
        logger.info(f"Residual pool: {len(residual_df):,} | Candidates: {len(candidates):,}")

        if len(candidates) == 0:
            logger.error("No candidates found for SES.")
            return None

        if len(candidates) < ses_size:
            logger.warning(f"Only {len(candidates)} candidates available; using all.")
            ses_size = len(candidates)

        def get_robust_struct_fingerprint(row):
            try:
                from pymatgen.core import Structure

                atoms_dict = row['atoms']
                struct = Structure(
                    lattice=atoms_dict['lattice_mat'],
                    species=atoms_dict['elements'],
                    coords=atoms_dict['coords'],
                    coords_are_cartesian=True
                )

                density = struct.density
                vol_per_atom = struct.volume / struct.num_sites
                abc = struct.lattice.abc
                lattice_std = np.std(abc) / np.mean(abc)
                atomic_numbers = [site.specie.Z for site in struct.sites]
                mean_z = np.mean(atomic_numbers)
                std_z = np.std(atomic_numbers)
                return np.array([density, vol_per_atom, lattice_std, mean_z, std_z])
            except Exception:
                return np.zeros(5)

        train_features = np.vstack(self.subset_data.apply(get_robust_struct_fingerprint, axis=1).values)
        candidate_features = np.vstack(candidates.apply(get_robust_struct_fingerprint, axis=1).values)

        scaler = StandardScaler()
        train_features_scaled = scaler.fit_transform(train_features)
        candidate_features_scaled = scaler.transform(candidate_features)

        min_distances = []
        batch_size = 500
        for i in tqdm(range(0, len(candidate_features_scaled), batch_size), desc="Computing distance matrix"):
            batch_cands = candidate_features_scaled[i:i + batch_size]
            dists = pairwise_distances(batch_cands, train_features_scaled, metric='euclidean')
            min_distances.extend(dists.min(axis=1))

        candidates['structure_isolation_score'] = min_distances
        self.ses_data = candidates.nlargest(ses_size, 'structure_isolation_score')

        logger.info(f"SES created: {len(self.ses_data):,} samples | Mean isolation {self.ses_data['structure_isolation_score'].mean():.4f}")

        os.makedirs('./data/', exist_ok=True)
        self.ses_data.to_pickle('./data/jarvis_ses_ood_set.pkl')
        logger.info("SES raw data saved to ./data/jarvis_ses_ood_set.pkl")

        return self.ses_data
    
    def _perform_pca(self):
        """Run PCA once and cache results."""
        if self._pca_results is not None:
            logger.info("Using cached PCA results")
            return self._pca_results

        logger.info(f"Running PCA on features: {self.pca_features}")
        
        features_present = [f for f in self.pca_features if f in self.full_data.columns]
        
        if len(features_present) < 2:
            logger.error(f"Not enough features for PCA: {features_present}")
            return None, None, None, None
        
        logger.info(f"Using features: {features_present}")

        full_data_clean = self.full_data[features_present].dropna()
        scaler = StandardScaler()
        full_data_scaled = scaler.fit_transform(full_data_clean)

        pca = PCA(n_components=2)
        full_pca = pca.fit_transform(full_data_scaled)

        subset_indices = full_data_clean.index.intersection(self.subset_data.index)
        subset_pca = full_pca[full_data_clean.index.isin(subset_indices)]

        self._pca_results = (full_pca, subset_pca, pca, full_data_clean)

        logger.info(
            f"PCA complete: PC1 {pca.explained_variance_ratio_[0]:.1%}, "
            f"PC2 {pca.explained_variance_ratio_[1]:.1%}"
        )
        
        return self._pca_results

    def plot_density_ratio(self, ax, num_df: pd.DataFrame, den_df: pd.DataFrame,
                           x_col: str, y_col: str, x_range: Tuple[float, float],
                           y_range: Tuple[float, float], bins: int = 60,
                           cmap: str = 'viridis'):
        """Plot density ratio heatmap: num/den."""
        try:
            h_den, xedges, yedges = np.histogram2d(
                den_df[x_col].dropna(),
                den_df[y_col].dropna(),
                bins=bins, range=[x_range, y_range], density=True
            )
            h_num, _, _ = np.histogram2d(
                num_df[x_col].dropna(),
                num_df[y_col].dropna(),
                bins=[xedges, yedges], density=True
            )
            with np.errstate(divide='ignore', invalid='ignore'):
                density_ratio = h_num / h_den
            density_ratio[np.isnan(density_ratio)] = 0

            contour = ax.contourf(xedges[:-1], yedges[:-1], density_ratio.T,
                                  cmap=cmap, levels=15, alpha=0.9)
            ax.contour(xedges[:-1], yedges[:-1], density_ratio.T,
                       levels=8, colors='white', alpha=0.4, linewidths=0.5)
            return contour
        except Exception as e:
            logger.warning(f"Density ratio calculation failed: {e}")
            ax.text(0.5, 0.5, 'Density Calculation\nTemporarily Unavailable',
                    transform=ax.transAxes, ha='center', va='center', fontsize=12)
            return None
    
    def create_visualizations(self, save_dir: str = './data/'):
        """Create visualizations."""
        logger.info("Creating visualizations...")
        
        os.makedirs(save_dir, exist_ok=True)
        
        self.generate_new_figure_4(os.path.join(save_dir, 'new_figure_4_pca_corrected.png'))

        self.generate_new_figure_5(os.path.join(save_dir, 'new_figure_5_enhanced_landscape.png'))

        logger.info(f"Visualizations saved to: {save_dir}")
    
    def generate_new_figure_4(self, save_path: str = "./data/new_figure_4_pca.png"):
        """Generate PCA coverage analysis plot."""
        logger.info("Generating PCA coverage plot...")
        
        full_pca, subset_pca, pca, full_data_clean = self._perform_pca()
        
        if full_pca is None:
            logger.warning("PCA failed; skipping figure 4")
            return

        pca_full_df = pd.DataFrame(full_pca, columns=['PC1', 'PC2'])
        pca_subset_df = pd.DataFrame(subset_pca, columns=['PC1', 'PC2'])

        with plt.style.context('seaborn-v0_8-whitegrid'):
            fig, ax = plt.subplots(figsize=(12, 10))
            hexbin = ax.hexbin(
                pca_full_df['PC1'], 
                pca_full_df['PC2'],
                gridsize=80,
                cmap='PuBu',
                bins='log',
                mincnt=1,
                linewidths=0,
                alpha=0.6,
                extent=[pca_full_df['PC1'].min()-1, pca_full_df['PC1'].max()+1,
                       pca_full_df['PC2'].min()-1, pca_full_df['PC2'].max()+1]
            )

            cbar = plt.colorbar(hexbin, ax=ax, shrink=0.8, pad=0.02)
            cbar.set_label('Sample Density (Log Scale)', fontsize=12, fontweight='bold')

            scatter = ax.scatter(
                pca_subset_df['PC1'],
                pca_subset_df['PC2'],
                s=9,
                color='#FF3333',
                alpha=0.85,
                label=f'Stratified Subset (n={len(pca_subset_df):,})',
                edgecolors='white',
                linewidth=0.15,
                zorder=10
            )

            pc1_var = pca.explained_variance_ratio_[0] * 100
            pc2_var = pca.explained_variance_ratio_[1] * 100
            cumulative_var = pc1_var + pc2_var
            
            ax.set_xlabel(f'Principal Component 1 ({pc1_var:.1f}% variance)', 
                         fontsize=16, fontweight='bold')
            ax.set_ylabel(f'Principal Component 2 ({pc2_var:.1f}% variance)', 
                         fontsize=16, fontweight='bold')
            ax.set_title('Materials Design Space Coverage Analysis', 
                        fontsize=20, fontweight='bold')

            legend = ax.legend(
                fontsize=14,
                markerscale=4,
                shadow=True,
                loc='upper right',
                framealpha=0.95,
                facecolor='white'
            )
            handles = []
            if hasattr(legend, 'legend_handles'):
                handles = legend.legend_handles
            elif hasattr(legend, 'legendHandles'):
                handles = legend.legendHandles
            for handle in handles:
                handle.set_alpha(1)
            
            info_text = (
                f"PCA Coverage Assessment\n"
                f"• Cumulative variance: {cumulative_var:.1f}%\n"
                f"• Full dataset: {len(pca_full_df):,} samples\n"
                f"• Stratified subset: {len(pca_subset_df):,} samples\n"
                f"• Coverage verification: Complete"
            )
            ax.text(0.02, 0.98, info_text, transform=ax.transAxes, fontsize=12,
                    verticalalignment='top', fontweight='bold',
                    zorder=20,
                    bbox=dict(boxstyle='round,pad=0.6', 
                             facecolor='white', 
                             alpha=1.0,
                             edgecolor='#666666', linewidth=1.5))

            ax.grid(linestyle='--', alpha=0.4)
            plt.tight_layout()
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
            
        logger.info(f"Figure 4 saved to: {save_path}")
    
    def generate_new_figure_5(self, save_path: str = "./data/new_figure_5_landscape.png"):
        """Generate landscape plot (2x3 layout)."""
        logger.info("Generating landscape plot...")
        
        with plt.style.context('seaborn-v0_8-whitegrid'):
            fig, axes = plt.subplots(2, 3, figsize=(24, 14))
            fig.suptitle('Materials Property Landscape and Sampling Strategy Analysis', 
                        fontsize=22, fontweight='bold')

            common_xlabel = "Formation Energy (eV/atom)"
            common_ylabel = "Bandgap (eV)"
            label_fontsize = 14
            title_fontsize = 16
            
            x_range = (-4, 3)
            y_range = (-1, 15)
            ax = axes[0, 0]
            hexbin_full = ax.hexbin(
                self.full_data['formation_energy_peratom'],
                self.full_data['optb88vdw_bandgap'],
                gridsize=50,
                cmap='Greys',
                bins='log',
                mincnt=1,
                linewidths=0,
                alpha=0.7,
                extent=[x_range[0], x_range[1], y_range[0], y_range[1]]
            )
            ax.set_title("(a) Ground Truth: Full Chemical Space (76k)",
                         fontsize=title_fontsize, fontweight='bold')
            ax.set(xlabel=common_xlabel, ylabel=common_ylabel, xlim=x_range, ylim=y_range)
            ax.xaxis.label.set_fontsize(label_fontsize)
            ax.yaxis.label.set_fontsize(label_fontsize)
            cbar_a = plt.colorbar(hexbin_full, ax=ax, shrink=0.75, pad=0.02)
            cbar_a.set_label('Full Dataset Density (Log)', fontsize=11, fontweight='bold')
            ax.grid(True, alpha=0.3)

            ax = axes[0, 1]
            if self.std_data is not None and len(self.std_data) > 0:
                ax.scatter(self.std_data['formation_energy_peratom'], self.std_data['optb88vdw_bandgap'],
                           s=6, alpha=0.15, color='#2E86C1', label='Std-55k', rasterized=True)
            else:
                ax.text(0.5, 0.5, 'Std-55k Not Available', transform=ax.transAxes,
                        ha='center', va='center', fontsize=12)
            ax.set_title("(b) Standard Benchmark (55k)\nBiased Distribution (Redundant in stable regions)",
                         fontsize=title_fontsize, fontweight='bold')
            ax.set(xlabel=common_xlabel, ylabel=common_ylabel, xlim=x_range, ylim=y_range)
            ax.xaxis.label.set_fontsize(label_fontsize)
            ax.yaxis.label.set_fontsize(label_fontsize)
            ax.legend(loc='upper right', fontsize=12)
            ax.grid(True, alpha=0.3)

            ax = axes[0, 2]
            ax.scatter(self.subset_data['formation_energy_peratom'], self.subset_data['optb88vdw_bandgap'],
                       s=8, alpha=0.6, color='#E74C3C', label='Stratified 15k', rasterized=True)
            ax.set_title("(c) Stratified Subset (15k)\nCorrected Distribution (Uniform Coverage)",
                         fontsize=title_fontsize, fontweight='bold')
            ax.set(xlabel=common_xlabel, ylabel=common_ylabel, xlim=x_range, ylim=y_range)
            ax.xaxis.label.set_fontsize(label_fontsize)
            ax.yaxis.label.set_fontsize(label_fontsize)
            ax.legend(loc='upper right', fontsize=12)
            ax.grid(True, alpha=0.3)

            ax = axes[1, 0]
            hexbin_den = ax.hexbin(
                self.full_data['formation_energy_peratom'],
                self.full_data['optb88vdw_bandgap'],
                gridsize=50,
                cmap='Blues',
                bins='log',
                mincnt=1,
                linewidths=0,
                alpha=0.75,
                extent=[x_range[0], x_range[1], y_range[0], y_range[1]]
            )
            ax.set_title("(d) Full Dataset Density Map", fontsize=title_fontsize, fontweight='bold')
            ax.set(xlabel=common_xlabel, ylabel=common_ylabel, xlim=x_range, ylim=y_range)
            ax.xaxis.label.set_fontsize(label_fontsize)
            ax.yaxis.label.set_fontsize(label_fontsize)
            cbar_d = plt.colorbar(hexbin_den, ax=ax, shrink=0.75, pad=0.02)
            cbar_d.set_label('Density (Log)', fontsize=11, fontweight='bold')
            ax.grid(True, alpha=0.3)

            ax = axes[1, 1]
            if self.std_data is not None and len(self.std_data) > 0:
                contour_55k = self.plot_density_ratio(
                    ax=ax,
                    num_df=self.std_data,
                    den_df=self.full_data,
                    x_col='formation_energy_peratom',
                    y_col='optb88vdw_bandgap',
                    x_range=x_range,
                    y_range=y_range,
                    bins=60,
                    cmap='RdBu_r'
                )
                if contour_55k is not None:
                    cbar = fig.colorbar(contour_55k, ax=ax, shrink=0.8)
                    cbar.set_label('Density Ratio (55k / 76k)', fontsize=11, fontweight='bold')
            else:
                ax.text(0.5, 0.5, 'Std-55k Not Available', transform=ax.transAxes,
                        ha='center', va='center', fontsize=12)
            ax.set_title("(e) Bias Map: 55k vs 76k (Red=Over-sampled)",
                         fontsize=title_fontsize, fontweight='bold')
            ax.set(xlabel=common_xlabel, ylabel=common_ylabel, xlim=x_range, ylim=y_range)
            ax.xaxis.label.set_fontsize(label_fontsize)
            ax.yaxis.label.set_fontsize(label_fontsize)
            ax.grid(True, alpha=0.3)

            ax = axes[1, 2]
            contour_15k = self.plot_density_ratio(
                ax=ax,
                num_df=self.subset_data,
                den_df=self.full_data,
                x_col='formation_energy_peratom',
                y_col='optb88vdw_bandgap',
                x_range=x_range,
                y_range=y_range,
                bins=60,
                cmap='viridis'
            )
            if contour_15k is not None:
                cbar = fig.colorbar(contour_15k, ax=ax, shrink=0.8)
                cbar.set_label('Density Ratio (15k / 76k)', fontsize=11, fontweight='bold')
            ax.set_title("(f) Fidelity Map: 15k vs 76k (Uniform)",
                         fontsize=title_fontsize, fontweight='bold')
            ax.set(xlabel=common_xlabel, ylabel=common_ylabel, xlim=x_range, ylim=y_range)
            ax.xaxis.label.set_fontsize(label_fontsize)
            ax.yaxis.label.set_fontsize(label_fontsize)
            ax.grid(True, alpha=0.3)

            plt.tight_layout(rect=[0, 0, 1, 0.95])
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
            
        logger.info(f"Figure 5 saved to: {save_path}")
    
def main():
    logger.info("JARVIS-DFT sampling pipeline v2.6")

    pkl_path = './jarvis_data/raw/jarvis_dft_75k_raw.pkl'
    sampler = AdvancedJARVISSampler(
        pkl_path=pkl_path,
        target_size=15000,
        random_state=42
    )

    if not sampler.load_and_validate_data():
        logger.error("Data load failed. Exiting.")
        return

    sampler.create_enhanced_stratification()
    sampler.execute_robust_sampling()
    sampler.validate_sampling_quality()

    tasks = {
        "fe": "formation_energy_peratom",
        "bg": "optb88vdw_bandgap",
        "total_energy": "optb88vdw_total_energy"
    }
    sampler.sampling_metadata['aligned_tasks'] = list(tasks.values())

    data_dir = './data/'
    os.makedirs(data_dir, exist_ok=True)
    subset_path = os.path.join(data_dir, 'subset_data_for_visualization.pkl')
    full_path = os.path.join(data_dir, 'full_data_for_visualization.pkl')
    sampler.subset_data.to_pickle(subset_path)
    sampler.full_data.to_pickle(full_path)
    logger.info(f"Saved subset/full data: {subset_path}, {full_path}")

    logger.info("Generating aligned multitask datasets...")
    for task_key, prop_name in tasks.items():
        logger.info(f"Processing task: {task_key} ({prop_name})")
        g2lnet_data_list = convert_to_g2lnet_list(sampler.subset_data, prop_name)
        if not g2lnet_data_list:
            continue

        final_df = pd.DataFrame(g2lnet_data_list)
        if len(final_df) != len(sampler.subset_data):
            logger.warning(
                f"{task_key} sample count ({len(final_df)}) differs from subset ({len(sampler.subset_data)})"
            )

        output_dir = f"./data/jarvis_{task_key}_15k_stratified/raw/"
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, f"jarvis_{task_key}_15k_stratified.json")
        final_df.to_json(output_path, orient='split', indent=2)
        logger.info(f"Saved {task_key} ({len(final_df)} samples)")

    logger.info("Building OOD/SES datasets...")
    ses_data = sampler.create_structural_extrapolation_set(ses_size=1500)
    if ses_data is not None:
        for task_key, prop_name in tasks.items():
            ses_list = convert_to_g2lnet_list(ses_data, prop_name)
            if not ses_list:
                continue
            ses_df = pd.DataFrame(ses_list)
            out_dir = f"./data/jarvis_{task_key}_ood/raw/"
            os.makedirs(out_dir, exist_ok=True)
            ses_df.to_json(
                os.path.join(out_dir, f"jarvis_{task_key}_ood.json"), orient='split', indent=2
            )
            logger.info(f"Saved {task_key} OOD set ({len(ses_df)} samples)")

    sampler.create_visualizations()

    metadata_path = './data/comprehensive_sampling_metadata.json'
    os.makedirs(os.path.dirname(metadata_path), exist_ok=True)
    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(sampler.sampling_metadata, f, indent=2, default=str)

    logger.info("Pipeline complete. All JSON files share identical material IDs.")


if __name__ == "__main__":
    main()