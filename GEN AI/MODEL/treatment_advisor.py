"""
Clinical Insight & Treatment Advisor Module.
Provides rule-based drug recommendations and treatment simulations based on tumor volume.
"""

from typing import Dict, List, Any

class TreatmentAdvisor:
    """Clinical decision support for lung tumor findings."""
    
    @staticmethod
    def calculate_volume(mask_sum: int, spacing: List[float]) -> float:
        """Calculate tumor volume in cubic centimeters (cc)."""
        voxel_vol = spacing[0] * spacing[1] * spacing[2]
        total_vol_mm3 = mask_sum * voxel_vol
        return total_vol_mm3 / 1000.0  # mm3 to cc

    @staticmethod
    def get_stage_and_recommendations(volume_cc: float) -> Dict[str, Any]:
        """Provide clinical insights based on tumor volume (TNM-T Proxy).
        
        Note: These are clinical-grade simulations for a University Final Year Project.
        """
        insights = {
            "volume_cc": round(volume_cc, 3),
            "stage_proxy": "T0",
            "risk_level": "None",
            "treatment_strategy": "Routine Follow-up",
            "recommended_drugs": []
        }
        
        if volume_cc <= 0:
            return insights

        if volume_cc < 1.0: # < 1cm3 roughly
            insights.update({
                "stage_proxy": "T1a (Stage I)",
                "risk_level": "Low",
                "treatment_strategy": "Active Surveillance or Stereotactic Radiotherapy (SBRT)",
                "recommended_drugs": ["Gefitinib (if EGFR+)", "Osimertinib"]
            })
        elif volume_cc < 10.0:
            insights.update({
                "stage_proxy": "T1c (Stage I-II)",
                "risk_level": "Moderate",
                "treatment_strategy": "Segmentectomy or Wedge Resection",
                "recommended_drugs": ["Afatinib", "Dacomitinib"]
            })
        elif volume_cc < 50.0:
            insights.update({
                "stage_proxy": "T2-T3 (Stage II-III)",
                "risk_level": "High",
                "treatment_strategy": "Lobectomy + Adjuvant Chemotherapy",
                "recommended_drugs": ["Cisplatin", "Paclitaxel", "Pembrolizumab (Immunotherapy)"]
            })
        else:
            insights.update({
                "stage_proxy": "T4 (Stage III-IV)",
                "risk_level": "Critical",
                "treatment_strategy": "Multimodal Therapy: Chemotherapy + Target Radiation",
                "recommended_drugs": ["Nivolumab", "Gemcitabine", "Bevacizumab"]
            })
            
        return insights

    @staticmethod
    def get_simulation_summary(volume_cc: float) -> str:
        """Generate a clinical summary for the demo report."""
        insights = TreatmentAdvisor.get_stage_and_recommendations(volume_cc)
        
        summary = f"CLINICAL FINDING REPORT\n"
        summary += f"------------------------\n"
        summary += f"Predicted Tumor Volume: {insights['volume_cc']} cc\n"
        summary += f"TNM Staging Proxy: {insights['stage_proxy']}\n"
        summary += f"Risk Category: {insights['risk_level']}\n\n"
        summary += f"PROPOSED TREATMENT SIMULATION:\n"
        summary += f"Primary Strategy: {insights['treatment_strategy']}\n"
        summary += f"Therapeutic Candidates: {', '.join(insights['recommended_drugs'])}\n"
        
        return summary
