import { useEffect, useState } from "react";
import { PatchCard } from "./PatchCard";
import { Patch, PatchMetrics } from "@/types/patch";

export function TopPatchesList() {
  const [patches, setPatches] = useState<Patch[]>([]);

  useEffect(() => {
    fetch("/api/top-patches")
      .then(res => res.json())
      .then(setPatches);
  }, []);

  return (
    <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
      {patches.map(patch => (
        <PatchCard
          key={patch.id}
          patch={patch}
          metrics={{
            id: patch.id,
            score: patch.score ?? 0,
            util_loss: patch.util_loss ?? 0,
            fp_rate: patch.fp_rate ?? 0,
            p95_ov: patch.p95_ov ?? 0
          }}
        />
      ))}
    </div>
  );
}
