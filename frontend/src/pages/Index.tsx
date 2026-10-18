import { useEffect, useState } from "react";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Badge } from "@/components/ui/badge";
import { PatchCard } from "@/components/PatchCard";
import { SemanticSearch } from "@/components/SemanticSearch";
import { MetricsDashboard } from "@/components/MetricsDashboard";
import { Shield, Settings, Search, BarChart3, RefreshCw, Plus } from "lucide-react";
import { Patch } from "@/types/patch";
import { TopPatchesList } from "@/components/TopPatchesList";

const API_URL = "http://127.0.0.1:8000/api/top-patches";

const Index = () => {
  const [patches, setPatches] = useState<Patch[]>([]);
  const [searchTerm, setSearchTerm] = useState("");

  useEffect(() => {
    fetch(API_URL)
      .then(res => res.json())
      .then(setPatches);
  }, []);

  const handleToggleActive = (patch: Patch, active: boolean) => {
    setPatches(prev => prev.map(p => 
      p.id === patch.id ? { ...p, active } : p
    ));
  };

  const handleViewDetails = (patch: Patch) => {
    // Show a modal with patch details
    alert(`Patch ID: ${patch.id}\n\nPrompt:\n${patch.prompt}`);
  };

  const filteredPatches = patches.filter(patch =>
    patch.id?.toLowerCase().includes(searchTerm.toLowerCase()) ||
    patch.prompt?.toLowerCase().includes(searchTerm.toLowerCase()) ||
    (patch.source || "").toLowerCase().includes(searchTerm.toLowerCase())
  );

  const activePatches = patches.filter(p => p.active);
  const totalPatches = patches.length;

  return (
    <div className="min-h-screen bg-background">
      {/* Header */}
      <header className="border-b border-border bg-card/50 backdrop-blur-sm">
        <div className="container mx-auto px-6 py-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-3">
              <div className="flex items-center justify-center w-10 h-10 rounded-lg bg-gradient-to-br from-primary to-primary/80">
                <Shield className="h-6 w-6 text-primary-foreground" />
              </div>
              <div>
                <h1 className="text-xl font-bold">Prompt Defender</h1>
                <p className="text-sm text-muted-foreground">Patch Management System</p>
              </div>
            </div>
            
            <div className="flex items-center gap-3">
              <Badge variant="outline" className="bg-success/10 text-success border-success">
                {activePatches.length}/{totalPatches} Active
              </Badge>
              <Button variant="outline" size="sm">
                <RefreshCw className="h-4 w-4 mr-2" />
                Refresh
              </Button>
              <Button size="sm">
                <Plus className="h-4 w-4 mr-2" />
                New Patch
              </Button>
            </div>
          </div>
        </div>
      </header>

      <div className="container mx-auto px-6 py-6">
        <Tabs defaultValue="patches" className="space-y-6">
          <TabsList className="grid w-full grid-cols-4 lg:w-400">
            <TabsTrigger value="patches" className="flex items-center gap-2">
              <Settings className="h-4 w-4" />
              Patches
            </TabsTrigger>
            <TabsTrigger value="search" className="flex items-center gap-2">
              <Search className="h-4 w-4" />
              Search
            </TabsTrigger>
            <TabsTrigger value="metrics" className="flex items-center gap-2">
              <BarChart3 className="h-4 w-4" />
              Metrics
            </TabsTrigger>
            <TabsTrigger value="deploy" className="flex items-center gap-2">
              <Shield className="h-4 w-4" />
              Deploy
            </TabsTrigger>
          </TabsList>

          <TabsContent value="patches" className="space-y-6">
            <div className="flex items-center justify-between">
              <div>
                <h2 className="text-2xl font-bold">Patch Catalog</h2>
                <p className="text-muted-foreground">Manage and monitor your security patches</p>
              </div>
              <div className="flex items-center gap-3">
                <Input
                  placeholder="Search patches..."
                  value={searchTerm}
                  onChange={(e) => setSearchTerm(e.target.value)}
                  className="w-64"
                />
              </div>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
              {filteredPatches.map((patch) => (
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
                  onToggleActive={handleToggleActive}
                  onViewDetails={handleViewDetails}
                />
              ))}
            </div>

            {filteredPatches.length === 0 && (
              <Card className="text-center py-8">
                <CardContent>
                  <p className="text-muted-foreground">No patches found matching your criteria</p>
                </CardContent>
              </Card>
            )}
          </TabsContent>

          <TabsContent value="search" className="space-y-6">
            <div>
              <h2 className="text-2xl font-bold mb-2">Semantic Patch Analysis</h2>
              <p className="text-muted-foreground">Find patches similar to incoming prompts using semantic embeddings</p>
            </div>
            
            <SemanticSearch patches={patches} />
          </TabsContent>

          <TabsContent value="metrics" className="space-y-6">
            <div>
              <h2 className="text-2xl font-bold mb-2">System Metrics</h2>
              <p className="text-muted-foreground">Real-time performance and safety metrics</p>
            </div>
            
            <MetricsDashboard metrics={{
              safety_pass_rate: 0,
              util_loss: 0,
              fp_rate: 0,
              latency: 0
            }} />
          </TabsContent>

          <TabsContent value="deploy" className="space-y-6">
            <div>
              <h2 className="text-2xl font-bold mb-2">Progressive Deployment</h2>
              <p className="text-muted-foreground">Manage canary rollouts and traffic distribution</p>
            </div>

            <Card>
              <CardHeader>
                <CardTitle>Deployment Pipeline</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="text-center py-8 text-muted-foreground">
                  <Shield className="h-12 w-12 mx-auto mb-4 opacity-50" />
                  <p>Progressive deployment controls coming soon...</p>
                  <p className="text-sm mt-2">Canary testing • Traffic splitting • Auto-rollback</p>
                </div>
              </CardContent>
            </Card>
          </TabsContent>
        </Tabs>

        {/* Top Patches Section */}
        <section className="my-8">
          <h2 className="text-xl font-bold mb-4">Top Patch Candidates</h2>
          <TopPatchesList />
        </section>
      </div>
    </div>
  );
};

export default Index;
