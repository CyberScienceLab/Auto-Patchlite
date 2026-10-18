import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Progress } from "@/components/ui/progress";
import { Badge } from "@/components/ui/badge";
import { LiveMetrics } from "@/types/patch";
import { TrendingUp, TrendingDown, Activity, Clock, Shield, AlertTriangle } from "lucide-react";

interface MetricsDashboardProps {
  metrics: LiveMetrics;
}

export function MetricsDashboard({ metrics }: MetricsDashboardProps) {
  const getMetricStatus = (value: number, threshold: number, isInverse = false) => {
    const condition = isInverse ? value < threshold : value > threshold;
    return condition ? "good" : "warning";
  };

  const MetricCard = ({ 
    title, 
    value, 
    threshold, 
    unit = "%", 
    icon: Icon, 
    isInverse = false,
    description 
  }: {
    title: string;
    value: number;
    threshold: number;
    unit?: string;
    icon: any;
    isInverse?: boolean;
    description: string;
  }) => {
    const status = getMetricStatus(value, threshold, isInverse);
    const percentage = unit === "%" ? value * 100 : value * 1000; // Convert latency to ms
    const displayValue = unit === "%" ? percentage.toFixed(1) : percentage.toFixed(0);
    const progressValue = unit === "%" ? percentage : Math.min(percentage / 2, 100); // Scale latency for progress bar

    return (
      <Card className={`border-l-4 ${
        status === "good" ? "border-l-success" : "border-l-warning"
      }`}>
        <CardHeader className="pb-2">
          <div className="flex items-center justify-between">
            <CardTitle className="text-sm font-medium text-muted-foreground">
              {title}
            </CardTitle>
            <Icon className={`h-4 w-4 ${
              status === "good" ? "text-success" : "text-warning"
            }`} />
          </div>
        </CardHeader>
        <CardContent className="space-y-3">
          <div className="flex items-baseline gap-2">
            <span className="text-2xl font-bold">
              {displayValue}
            </span>
            <span className="text-sm text-muted-foreground">
              {unit === "%" ? "%" : "ms"}
            </span>
            <Badge 
              variant={status === "good" ? "default" : "destructive"}
              className="ml-auto text-xs"
            >
              {status === "good" ? "Within SLO" : "Above SLO"}
            </Badge>
          </div>
          
          <div className="space-y-1">
            <div className="flex justify-between text-xs">
              <span className="text-muted-foreground">Target</span>
              <span className="text-muted-foreground">
                {isInverse ? ">" : "<"} {(threshold * (unit === "%" ? 100 : 1000)).toFixed(unit === "%" ? 1 : 0)}{unit === "%" ? "%" : "ms"}
              </span>
            </div>
            <Progress 
              value={progressValue} 
              className={`h-2 ${
                status === "good" ? "bg-success/20" : "bg-destructive/20"
              }`}
            />
          </div>
          
          <p className="text-xs text-muted-foreground">
            {description}
          </p>
        </CardContent>
      </Card>
    );
  };

  return (
    <div className="space-y-4">
      <div className="flex items-center gap-2 mb-4">
        <Activity className="h-5 w-5 text-primary" />
        <h3 className="text-lg font-semibold">Live Metrics Dashboard</h3>
        <Badge variant="outline" className="bg-primary/10 text-primary border-primary">
          Real-time
        </Badge>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
        <MetricCard
          title="Safety Pass Rate"
          value={metrics.safety_pass_rate}
          threshold={0.95}
          icon={Shield}
          isInverse={true}
          description="Percentage of malicious prompts successfully blocked"
        />
        
        <MetricCard
          title="Utility Loss"
          value={metrics.util_loss}
          threshold={0.02}
          icon={TrendingDown}
          description="Rate of legitimate requests incorrectly blocked"
        />
        
        <MetricCard
          title="False Positive Rate"
          value={metrics.fp_rate}
          threshold={0.01}
          icon={AlertTriangle}
          description="Benign prompts incorrectly flagged as malicious"
        />
        
        <MetricCard
          title="Latency"
          value={metrics.latency}
          threshold={0.15}
          unit="s"
          icon={Clock}
          description="Average response time for patch processing"
        />
      </div>

      <Card>
        <CardHeader>
          <CardTitle className="text-sm">System Health Summary</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-2 gap-4 text-sm">
            <div>
              <span className="text-muted-foreground">Overall Status:</span>
              <Badge className="ml-2 bg-success text-success-foreground">
                Operational
              </Badge>
            </div>
            <div>
              <span className="text-muted-foreground">Active Patches:</span>
              <span className="ml-2 font-mono">3/4</span>
            </div>
            <div>
              <span className="text-muted-foreground">Total Traffic:</span>
              <span className="ml-2 font-mono">1.2M req/hr</span>
            </div>
            <div>
              <span className="text-muted-foreground">Last Updated:</span>
              <span className="ml-2 font-mono">2s ago</span>
            </div>
          </div>
        </CardContent>
      </Card>
    </div>
  );
}