import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Switch } from "@/components/ui/switch";
import { Patch, PatchMetrics } from "@/types/patch";
import { Play, Pause, Eye, TrendingUp, AlertTriangle, CheckCircle } from "lucide-react";
import { useState } from "react";

interface PatchCardProps {
  patch: Patch;
  metrics?: PatchMetrics;
  onToggleActive?: (patch: Patch, active: boolean) => void;
  onViewDetails?: (patch: Patch) => void;
  similarity?: number;
}

export function PatchCard({ patch, metrics, onToggleActive, onViewDetails, similarity }: PatchCardProps) {
  const [isActive, setIsActive] = useState(patch.active || false);

  const handleToggle = () => {
    const newActive = !isActive;
    setIsActive(newActive);
    onToggleActive?.(patch, newActive);
  };

  const getStatusBadge = () => {
    switch (patch.status) {
      case "deployed":
        return <Badge variant="default" className="bg-success text-success-foreground">Deployed</Badge>;
      case "canary":
        return <Badge variant="secondary" className="bg-warning text-warning-foreground">Canary</Badge>;
      case "candidate":
        return <Badge variant="outline">Candidate</Badge>;
      default:
        return <Badge variant="secondary">Unknown</Badge>;
    }
  };

  const getSourceIcon = () => {
    switch (patch.source) {
      case "garak":
        return <AlertTriangle className="h-4 w-4 text-warning" />;
      case "inference":
        return <TrendingUp className="h-4 w-4 text-primary" />;
      case "manual":
        return <CheckCircle className="h-4 w-4 text-success" />;
      default:
        return null;
    }
  };

  return (
    <Card className="hover:shadow-lg transition-all duration-200 border-border bg-card">
      <CardHeader className="pb-3">
        <div className="flex items-start justify-between">
          <div className="space-y-2">
            <CardTitle className="text-lg flex items-center gap-2">
              <span className="font-mono text-primary">{patch.id}</span>
              {getSourceIcon()}
            </CardTitle>
            <div className="flex items-center gap-2">
              {getStatusBadge()}
              {patch.traffic_fraction !== undefined && (
                <Badge variant="outline" className="text-xs">
                  {Math.round(patch.traffic_fraction * 100)}% Traffic
                </Badge>
              )}
              {similarity !== undefined && (
                <Badge variant="secondary" className="text-xs">
                  {Math.round(similarity * 100)}% Similar
                </Badge>
              )}
            </div>
          </div>
          <Switch
            checked={isActive}
            onCheckedChange={handleToggle}
            className="data-[state=checked]:bg-primary"
          />
        </div>
      </CardHeader>
      
      <CardContent className="space-y-4">
        <div>
          <p className="text-sm text-muted-foreground mb-2">Prompt Preview:</p>
          <div className="text-sm bg-muted p-2 rounded text-foreground" style={{ maxHeight: 80, overflowY: 'auto', wordBreak: 'break-word' }}>
            {patch.prompt}
          </div>
        </div>

        {metrics && (
          <div className="grid grid-cols-2 gap-2 text-xs">
            <div className="space-y-1">
              <div className="flex justify-between">
                <span className="text-muted-foreground">Score:</span>
                <span className="font-mono text-foreground">{(metrics.score * 100).toFixed(1)}%</span>
              </div>
              <div className="flex justify-between">
                <span className="text-muted-foreground">FP Rate:</span>
                <span className="font-mono text-foreground">{(metrics.fp_rate * 100).toFixed(1)}%</span>
              </div>
            </div>
            <div className="space-y-1">
              <div className="flex justify-between">
                <span className="text-muted-foreground">Util Loss:</span>
                <span className="font-mono text-foreground">{(metrics.util_loss * 100).toFixed(1)}%</span>
              </div>
              <div className="flex justify-between">
                <span className="text-muted-foreground">P95 OV:</span>
                <span className="font-mono text-foreground">{(metrics.p95_ov * 100).toFixed(1)}%</span>
              </div>
            </div>
          </div>
        )}

        <div className="flex gap-2 pt-2">
          <Button
            size="sm"
            variant="outline"
            onClick={() => onViewDetails?.(patch)}
            className="flex-1"
          >
            <Eye className="h-4 w-4 mr-2" />
            Details
          </Button>
          {isActive ? (
            <Button size="sm" variant="destructive" onClick={handleToggle}>
              <Pause className="h-4 w-4 mr-2" />
              Pause
            </Button>
          ) : (
            <Button size="sm" variant="default" onClick={handleToggle}>
              <Play className="h-4 w-4 mr-2" />
              Deploy
            </Button>
          )}
        </div>
      </CardContent>
    </Card>
  );
}