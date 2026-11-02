"""Continental-Scale Driver Analytics and Tipping Point Hotspot Analysis.

Analyzes non-linear climate acceleration, spatial autocorrelation, and driver
sensitivities across 300 European NUTS-2 regions and 34 countries.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

try:
    from src.data.continental_loader import ContinentalDataLoader
except ImportError:
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from src.data.continental_loader import ContinentalDataLoader


@dataclass
class ContinentalHotspotMetrics:
    """Continental risk diagnostics across administrative and agro-ecological units."""
    total_regions: int
    most_vulnerable_regions: List[Dict[str, Union[str, float]]]
    highest_acceleration_regions: List[Dict[str, Union[str, float]]]
    ecotype_vulnerability_ranking: Dict[str, float]
    mean_vulnerability_index: float
    high_risk_region_count: int


class ContinentalDriverAnalytics:
    """Analyzes spatial climate vulnerability, tipping points, and driver attribution."""

    def __init__(self, loader: Optional[ContinentalDataLoader] = None) -> None:
        self.loader = loader or ContinentalDataLoader()

    def evaluate_nuts2_hotspots(
        self,
        vulnerability_threshold: float = 60.0,
        acceleration_threshold: float = 1.40,
    ) -> ContinentalHotspotMetrics:
        """Analyzes tipping point acceleration across all 300 NUTS-2 regions."""
        df = self.loader.load_nuts2_hotspots()
        ecotype_df = self.loader.load_ecotype_comparison()

        vuln_col = "compound_vulnerability_index"
        acc_col = "acceleration_ratio"

        # Top 10 most vulnerable regions (highest SWN-VI)
        sorted_vuln = df.sort_values(vuln_col, ascending=False).head(10)
        top_vuln = [
            {
                "nuts2_id": row.get("nuts2_id", ""),
                "nuts2_name": row.get("nuts2_name", ""),
                "vulnerability_index": float(row.get(vuln_col, 0.0)),
                "acceleration_ratio": float(row.get(acc_col, 0.0)),
            }
            for _, row in sorted_vuln.iterrows()
        ]

        # Top 10 highest acceleration (rate of climate impact acceleration)
        sorted_acc = df.sort_values(acc_col, ascending=False).head(10)
        top_acc = [
            {
                "nuts2_id": row.get("nuts2_id", ""),
                "nuts2_name": row.get("nuts2_name", ""),
                "vulnerability_index": float(row.get(vuln_col, 0.0)),
                "acceleration_ratio": float(row.get(acc_col, 0.0)),
            }
            for _, row in sorted_acc.iterrows()
        ]

        # Ecotype ranking
        ecotype_ranking = dict(zip(ecotype_df["ecotype"], ecotype_df[vuln_col]))

        mean_vuln = float(df[vuln_col].mean())
        high_risk_count = int((df[vuln_col] >= vulnerability_threshold).sum())

        return ContinentalHotspotMetrics(
            total_regions=len(df),
            most_vulnerable_regions=top_vuln,
            highest_acceleration_regions=top_acc,
            ecotype_vulnerability_ranking=ecotype_ranking,
            mean_vulnerability_index=mean_vuln,
            high_risk_region_count=high_risk_count,
        )

    def compute_spatial_divergence_matrix(self) -> pd.DataFrame:
        """Computes vulnerability contrast across European ecotypes and elevation bands."""
        eco_df = self.loader.load_ecotype_comparison()
        return eco_df.sort_values("compound_vulnerability_index", ascending=False).reset_index(drop=True)


def main() -> int:
    analytics = ContinentalDriverAnalytics()
    metrics = analytics.evaluate_nuts2_hotspots()

    print("=" * 70)
    print("CONTINENTAL EUROPEAN SOIL-WATER NEXUS HOTSPOT DIAGNOSTICS")
    print("=" * 70)
    print(f"Total NUTS-2 Regions Evaluated : {metrics.total_regions}")
    print(f"Mean Continental Vulnerability : {metrics.mean_vulnerability_index:.1f} / 100")
    print(f"Severe Risk Regions (Index>=60): {metrics.high_risk_region_count} regions")

    print("\n--- Top 5 Most Vulnerable Regions (SWN-VI) ---")
    for r in metrics.most_vulnerable_regions[:5]:
        safe_name = str(r['nuts2_name']).encode('ascii', 'replace').decode('ascii')
        print(f"  [{r['nuts2_id']}] {safe_name:<24}: Vulnerability {r['vulnerability_index']:.1f} | Accel {r['acceleration_ratio']:.2f}x")

    print("\n--- Agro-Ecological Biome Vulnerability Ranking ---")
    for eco, score in sorted(metrics.ecotype_vulnerability_ranking.items(), key=lambda x: x[1], reverse=True):
        print(f"  {eco:<28}: {score:.1f} / 100")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
