"""
Fusion Layer: Adaptive Confidence-Based Fusion for NLP + SLM

This module implements the fusion logic for combining Deberta (NLP) and Gemma (SLM)
predictions without requiring any training. The fusion uses confidence-based adaptive
weighting to intelligently combine both models' outputs.

Features:
- Adaptive weighting based on model confidence
- No training required (pure mathematical combination)
- Maintains interpretability and transparency
- Supports privacy-aware local deployment
"""

import numpy as np
from typing import Dict, Any, List, Optional


class AdaptiveFusion:
    """
    Adaptive Confidence-Based Fusion for combining NLP and SLM predictions.
    
    This fusion mechanism dynamically weights model predictions based on their
    confidence levels, giving more weight to confident predictions and less to
    uncertain ones.
    
    Args:
        deberta_base_weight (float): Base weight for Deberta when confidences are equal.
                                     Default: 0.7 (since Deberta has 97.91% accuracy)
        gemma_base_weight (float): Base weight for Gemma. Default: 0.3
        min_confidence_threshold (float): Minimum confidence to trust a prediction
    """
    
    def __init__(
        self,
        deberta_base_weight: float = 0.7,
        gemma_base_weight: float = 0.3,
        min_confidence_threshold: float = 0.1
    ):
        assert abs(deberta_base_weight + gemma_base_weight - 1.0) < 1e-6, \
            "Base weights must sum to 1.0"
        
        self.deberta_base_weight = deberta_base_weight
        self.gemma_base_weight = gemma_base_weight
        self.min_confidence_threshold = min_confidence_threshold
    
    def compute_confidence(self, probability: float) -> float:
        """
        Compute confidence score from a probability.
        
        Confidence is measured as distance from 0.5 (maximum uncertainty).
        A prediction of 0.95 or 0.05 both have high confidence (0.9).
        A prediction of 0.5 has zero confidence (completely uncertain).
        
        Args:
            probability: Probability value between 0 and 1
            
        Returns:
            Confidence score between 0 and 1
        """
        return abs(probability - 0.5) * 2
    
    def fuse(
        self,
        deberta_prob: float,
        gemma_prob: float,
        return_details: bool = False
    ) -> Dict[str, Any]:
        """
        Fuse predictions from Deberta and Gemma using adaptive weighting.
        
        Args:
            deberta_prob: Phishing probability from Deberta (0-1)
            gemma_prob: Phishing probability from Gemma (0-1)
            return_details: If True, return detailed breakdown of fusion process
            
        Returns:
            Dictionary containing:
                - probability: Final fused probability
                - label: Binary classification ('phishing' or 'safe')
                - confidence: Overall confidence in the prediction
                - weights: Individual model weights used (if return_details=True)
                - contributions: Individual model contributions (if return_details=True)
        """
        # Compute individual confidences
        conf_deberta = self.compute_confidence(deberta_prob)
        conf_gemma = self.compute_confidence(gemma_prob)
        
        # Calculate adaptive weights
        total_conf = conf_deberta + conf_gemma
        
        if total_conf > self.min_confidence_threshold:
            # Both models have some confidence: weight by relative confidence
            w_deberta = conf_deberta / total_conf
            w_gemma = conf_gemma / total_conf
        else:
            # Both models very uncertain: use base weights
            w_deberta = self.deberta_base_weight
            w_gemma = self.gemma_base_weight
        
        # Compute fused probability
        final_prob = (w_deberta * deberta_prob) + (w_gemma * gemma_prob)
        
        # Determine label
        label = 'phishing' if final_prob > 0.5 else 'safe'
        
        # Overall confidence (max of the two, since we're combining)
        overall_confidence = max(conf_deberta, conf_gemma)
        
        result = {
            'probability': final_prob,
            'label': label,
            'confidence': overall_confidence
        }
        
        if return_details:
            result['weights'] = {
                'deberta': w_deberta,
                'gemma': w_gemma
            }
            result['individual_probabilities'] = {
                'deberta': deberta_prob,
                'gemma': gemma_prob
            }
            result['individual_confidences'] = {
                'deberta': conf_deberta,
                'gemma': conf_gemma
            }
            result['fusion_strategy'] = (
                'confidence_based' if total_conf > self.min_confidence_threshold 
                else 'base_weights'
            )
        
        return result
    
    def fuse_with_explanation(
        self,
        deberta_prob: float,
        gemma_prob: float,
        gemma_explanation: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Fuse predictions and combine with Gemma's explanation.
        
        Args:
            deberta_prob: Phishing probability from Deberta
            gemma_prob: Phishing probability from Gemma
            gemma_explanation: Optional explanation dict from Gemma containing:
                - explanation: Text explanation
                - tactics: List of detected tactics
                - evidence: List of evidence items
                
        Returns:
            Complete result with classification and interpretable explanation
        """
        # Get fusion result with details
        fusion_result = self.fuse(deberta_prob, gemma_prob, return_details=True)
        
        # Add explanation if provided
        if gemma_explanation:
            fusion_result['explanation'] = gemma_explanation.get('explanation', '')
            fusion_result['tactics'] = gemma_explanation.get('tactics', [])
            fusion_result['evidence'] = gemma_explanation.get('evidence', [])
        
        # Add transparency information
        fusion_result['transparency'] = self._generate_transparency_info(
            fusion_result
        )
        
        return fusion_result
    
    def _generate_transparency_info(self, fusion_result: Dict[str, Any]) -> Dict[str, str]:
        """
        Generate human-readable transparency information about the fusion.
        
        Args:
            fusion_result: Result dictionary from fusion
            
        Returns:
            Dictionary with transparency explanations
        """
        w_d = fusion_result['weights']['deberta']
        w_g = fusion_result['weights']['gemma']
        strategy = fusion_result['fusion_strategy']
        
        if strategy == 'confidence_based':
            if w_d > 0.6:
                primary_model = "Deberta (NLP)"
                weight_pct = f"{w_d:.0%}"
            elif w_g > 0.6:
                primary_model = "Gemma (SLM)"
                weight_pct = f"{w_g:.0%}"
            else:
                primary_model = "balanced combination"
                weight_pct = "~50%"
            
            summary = (
                f"Classification based on {primary_model} ({weight_pct} weight) "
                f"due to higher confidence. "
            )
        else:
            summary = (
                f"Both models uncertain, using base weights "
                f"(Deberta: {w_d:.0%}, Gemma: {w_g:.0%}). "
            )
        
        return {
            'summary': summary,
            'deberta_weight': f"{w_d:.2f}",
            'gemma_weight': f"{w_g:.2f}",
            'fusion_strategy': strategy
        }
    
    def batch_fuse(
        self,
        deberta_probs: List[float],
        gemma_probs: List[float]
    ) -> List[Dict[str, Any]]:
        """
        Fuse predictions for multiple samples in batch.
        
        Args:
            deberta_probs: List of Deberta probabilities
            gemma_probs: List of Gemma probabilities
            
        Returns:
            List of fusion results
        """
        assert len(deberta_probs) == len(gemma_probs), \
            "Deberta and Gemma probability lists must have same length"
        
        results = []
        for d_prob, g_prob in zip(deberta_probs, gemma_probs):
            results.append(self.fuse(d_prob, g_prob))
        
        return results


class WeightedEnsemble:
    """
    Simple weighted ensemble as baseline comparison.
    
    Uses fixed weights (no adaptation based on confidence).
    """
    
    def __init__(self, deberta_weight: float = 0.7):
        self.deberta_weight = deberta_weight
        self.gemma_weight = 1 - deberta_weight
    
    def fuse(self, deberta_prob: float, gemma_prob: float) -> Dict[str, Any]:
        """Simple weighted average fusion."""
        final_prob = (self.deberta_weight * deberta_prob + 
                     self.gemma_weight * gemma_prob)
        
        return {
            'probability': final_prob,
            'label': 'phishing' if final_prob > 0.5 else 'safe',
            'weights': {
                'deberta': self.deberta_weight,
                'gemma': self.gemma_weight
            }
        }


def compare_fusion_methods(
    deberta_prob: float,
    gemma_prob: float
) -> Dict[str, Dict[str, Any]]:
    """
    Compare different fusion methods on a single example.
    
    Useful for analysis and understanding fusion behavior.
    
    Args:
        deberta_prob: Deberta prediction
        gemma_prob: Gemma prediction
        
    Returns:
        Dictionary comparing different fusion strategies
    """
    adaptive = AdaptiveFusion()
    weighted = WeightedEnsemble()
    
    return {
        'adaptive_fusion': adaptive.fuse(deberta_prob, gemma_prob, return_details=True),
        'weighted_ensemble': weighted.fuse(deberta_prob, gemma_prob),
        'deberta_only': {
            'probability': deberta_prob,
            'label': 'phishing' if deberta_prob > 0.5 else 'safe'
        },
        'gemma_only': {
            'probability': gemma_prob,
            'label': 'phishing' if gemma_prob > 0.5 else 'safe'
        }
    }


if __name__ == "__main__":
    # Example usage and demonstration
    print("=" * 60)
    print("Adaptive Fusion Demonstration")
    print("=" * 60)
    
    fusion = AdaptiveFusion()
    
    # Test Case 1: Both models confident and agree
    print("\n1. Both confident, both agree (clear phishing):")
    result = fusion.fuse(0.95, 0.88, return_details=True)
    print(f"   Deberta: 0.95, Gemma: 0.88")
    print(f"   → Fused: {result['probability']:.3f} ({result['label']})")
    print(f"   → Weights: Deberta={result['weights']['deberta']:.2f}, "
          f"Gemma={result['weights']['gemma']:.2f}")
    
    # Test Case 2: Deberta confident, Gemma uncertain
    print("\n2. Deberta confident, Gemma uncertain:")
    result = fusion.fuse(0.92, 0.55, return_details=True)
    print(f"   Deberta: 0.92, Gemma: 0.55")
    print(f"   → Fused: {result['probability']:.3f} ({result['label']})")
    print(f"   → Weights: Deberta={result['weights']['deberta']:.2f}, "
          f"Gemma={result['weights']['gemma']:.2f}")
    
    # Test Case 3: Both uncertain
    print("\n3. Both uncertain (edge case):")
    result = fusion.fuse(0.52, 0.48, return_details=True)
    print(f"   Deberta: 0.52, Gemma: 0.48")
    print(f"   → Fused: {result['probability']:.3f} ({result['label']})")
    print(f"   → Weights: Deberta={result['weights']['deberta']:.2f}, "
          f"Gemma={result['weights']['gemma']:.2f}")
    print(f"   → Strategy: {result['fusion_strategy']}")
    
    # Test Case 4: Models disagree
    print("\n4. Models disagree (Deberta says safe, Gemma says phishing):")
    result = fusion.fuse(0.35, 0.75, return_details=True)
    print(f"   Deberta: 0.35, Gemma: 0.75")
    print(f"   → Fused: {result['probability']:.3f} ({result['label']})")
    print(f"   → Weights: Deberta={result['weights']['deberta']:.2f}, "
          f"Gemma={result['weights']['gemma']:.2f}")
    
    print("\n" + "=" * 60)
