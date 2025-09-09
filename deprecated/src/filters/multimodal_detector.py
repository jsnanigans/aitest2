"""
Multimodal detection to identify when multiple people are using the same account.
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
from collections import defaultdict
import logging
from sklearn.mixture import GaussianMixture
from sklearn.cluster import DBSCAN

logger = logging.getLogger(__name__)

class MultimodalDetector:
    def __init__(self, config: Dict = None):
        self.config = config or {}
        
        self.min_readings_for_detection = self.config.get('min_readings_for_detection', 30)
        self.cluster_gap_kg = self.config.get('cluster_gap_kg', 15.0)
        self.min_cluster_size = self.config.get('min_cluster_size', 5)
        self.max_components = self.config.get('max_components', 3)
        self.enable_auto_split = self.config.get('enable_auto_split', True)
        
        self.user_clusters = {}
        self.detection_results = {}
    
    def detect_multimodal(self, user_id: str, weights: List[float]) -> Dict:
        """
        Detect if weight distribution is multimodal (multiple people).
        Returns detection results including number of clusters and assignments.
        """
        
        if len(weights) < self.min_readings_for_detection:
            return {
                'is_multimodal': False,
                'reason': 'Insufficient data',
                'num_clusters': 1
            }
        
        weights_array = np.array(weights).reshape(-1, 1)
        
        method1_result = self._detect_via_gap_analysis(weights)
        
        method2_result = self._detect_via_gaussian_mixture(weights_array)
        
        if method1_result['is_multimodal'] or method2_result['is_multimodal']:
            primary_result = method1_result if method1_result['is_multimodal'] else method2_result
            
            cluster_assignments = self._assign_to_clusters(weights, primary_result['cluster_centers'])
            
            result = {
                'is_multimodal': True,
                'num_clusters': primary_result['num_clusters'],
                'cluster_centers': primary_result['cluster_centers'],
                'cluster_stats': self._calculate_cluster_stats(weights, cluster_assignments),
                'assignments': cluster_assignments,
                'detection_method': primary_result['method']
            }
            
            self.detection_results[user_id] = result
            
            if self.enable_auto_split:
                self._create_virtual_users(user_id, weights, cluster_assignments)
            
            return result
        
        return {
            'is_multimodal': False,
            'num_clusters': 1,
            'reason': 'Single distribution detected'
        }
    
    def _detect_via_gap_analysis(self, weights: List[float]) -> Dict:
        """
        Simple gap-based clustering: find large gaps in sorted weights.
        """
        sorted_weights = sorted(weights)
        clusters = []
        current_cluster = [sorted_weights[0]]
        
        for i in range(1, len(sorted_weights)):
            if sorted_weights[i] - current_cluster[-1] > self.cluster_gap_kg:
                if len(current_cluster) >= self.min_cluster_size:
                    clusters.append(current_cluster)
                current_cluster = [sorted_weights[i]]
            else:
                current_cluster.append(sorted_weights[i])
        
        if len(current_cluster) >= self.min_cluster_size:
            clusters.append(current_cluster)
        
        if len(clusters) > 1:
            cluster_centers = [np.mean(c) for c in clusters]
            return {
                'is_multimodal': True,
                'num_clusters': len(clusters),
                'cluster_centers': cluster_centers,
                'method': 'gap_analysis'
            }
        
        return {'is_multimodal': False}
    
    def _detect_via_gaussian_mixture(self, weights_array: np.ndarray) -> Dict:
        """
        Use Gaussian Mixture Model to detect multimodal distributions.
        """
        best_n_components = 1
        best_bic = float('inf')
        
        for n_components in range(1, min(self.max_components + 1, len(weights_array) // 10)):
            try:
                gmm = GaussianMixture(
                    n_components=n_components,
                    covariance_type='full',
                    random_state=42
                )
                gmm.fit(weights_array)
                bic = gmm.bic(weights_array)
                
                if bic < best_bic:
                    best_bic = bic
                    best_n_components = n_components
                    best_gmm = gmm
            except:
                continue
        
        if best_n_components > 1:
            cluster_centers = best_gmm.means_.flatten().tolist()
            
            cluster_stds = np.sqrt(best_gmm.covariances_).flatten()
            for i in range(len(cluster_centers) - 1):
                separation = abs(cluster_centers[i+1] - cluster_centers[i])
                combined_std = cluster_stds[i] + cluster_stds[i+1]
                if separation < combined_std * 2:
                    return {'is_multimodal': False}
            
            return {
                'is_multimodal': True,
                'num_clusters': best_n_components,
                'cluster_centers': sorted(cluster_centers),
                'method': 'gaussian_mixture'
            }
        
        return {'is_multimodal': False}
    
    def _assign_to_clusters(self, weights: List[float], cluster_centers: List[float]) -> List[int]:
        """
        Assign each weight to nearest cluster center.
        """
        assignments = []
        for weight in weights:
            distances = [abs(weight - center) for center in cluster_centers]
            assignments.append(distances.index(min(distances)))
        return assignments
    
    def _calculate_cluster_stats(self, weights: List[float], assignments: List[int]) -> List[Dict]:
        """
        Calculate statistics for each cluster.
        """
        clusters = defaultdict(list)
        for weight, assignment in zip(weights, assignments):
            clusters[assignment].append(weight)
        
        stats = []
        for cluster_id in sorted(clusters.keys()):
            cluster_weights = clusters[cluster_id]
            stats.append({
                'cluster_id': cluster_id,
                'count': len(cluster_weights),
                'mean': np.mean(cluster_weights),
                'std': np.std(cluster_weights),
                'min': min(cluster_weights),
                'max': max(cluster_weights),
                'median': np.median(cluster_weights)
            })
        
        return stats
    
    def _create_virtual_users(self, user_id: str, weights: List[float], assignments: List[int]):
        """
        Create virtual user profiles for each detected cluster.
        """
        virtual_users = defaultdict(list)
        
        for weight, cluster_id in zip(weights, assignments):
            virtual_user_id = f"{user_id}_cluster_{cluster_id}"
            virtual_users[virtual_user_id].append(weight)
        
        self.user_clusters[user_id] = virtual_users
        
        logger.info(f"Created {len(virtual_users)} virtual users for {user_id}")
        for vid, vweights in virtual_users.items():
            logger.info(f"  {vid}: {len(vweights)} readings, mean={np.mean(vweights):.1f}kg")
    
    def get_virtual_user_id(self, user_id: str, weight: float) -> str:
        """
        Get the appropriate virtual user ID for a given weight.
        """
        if user_id not in self.detection_results:
            return user_id
        
        result = self.detection_results[user_id]
        if not result['is_multimodal']:
            return user_id
        
        distances = [abs(weight - center) for center in result['cluster_centers']]
        cluster_id = distances.index(min(distances))
        
        return f"{user_id}_cluster_{cluster_id}"
    
    def get_detection_summary(self) -> Dict:
        """
        Get summary of all multimodal detections.
        """
        summary = {
            'total_users_analyzed': len(self.detection_results),
            'multimodal_users': sum(1 for r in self.detection_results.values() if r['is_multimodal']),
            'virtual_users_created': sum(len(vu) for vu in self.user_clusters.values()),
            'details': {}
        }
        
        for user_id, result in self.detection_results.items():
            if result['is_multimodal']:
                summary['details'][user_id] = {
                    'num_clusters': result['num_clusters'],
                    'cluster_stats': result.get('cluster_stats', []),
                    'method': result.get('detection_method', 'unknown')
                }
        
        return summary