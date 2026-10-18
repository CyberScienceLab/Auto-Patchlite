import { useState } from "react";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Textarea } from "@/components/ui/textarea";
import { Badge } from "@/components/ui/badge";
import { Search, AlertCircle } from "lucide-react";
import { Patch, PatchSimilarity } from "@/types/patch";
import { mockPatches, cosineSimilarity, computeMockEmbedding } from "@/lib/mockData";

interface SemanticSearchProps {
  patches?: Patch[];
}

export function SemanticSearch({ patches = mockPatches }: SemanticSearchProps) {
  const [query, setQuery] = useState("");
  const [results, setResults] = useState<PatchSimilarity[]>([]);
  const [isSearching, setIsSearching] = useState(false);

  const handleSearch = () => {
    if (!query.trim()) return;
    
    setIsSearching(true);
    
    // Simulate async processing
    setTimeout(() => {
      const queryEmbedding = computeMockEmbedding(query);
      
      const similarities = patches
        .map(patch => ({
          patch,
          similarity: cosineSimilarity(queryEmbedding, patch.embedding.slice(0, 10))
        }))
        .filter(item => item.similarity > 0.1) // Filter very low similarity
        .sort((a, b) => b.similarity - a.similarity)
        .slice(0, 5); // Top 5 results
      
      setResults(similarities);
      setIsSearching(false);
    }, 500);
  };

  const getSimilarityColor = (similarity: number) => {
    if (similarity > 0.8) return "bg-destructive text-destructive-foreground";
    if (similarity > 0.6) return "bg-warning text-warning-foreground";
    if (similarity > 0.4) return "bg-primary text-primary-foreground";
    return "bg-muted text-muted-foreground";
  };

  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          <Search className="h-5 w-5" />
          Semantic Patch Search
        </CardTitle>
      </CardHeader>
      <CardContent className="space-y-4">
        <div className="space-y-2">
          <Textarea
            placeholder="Enter a prompt to find similar patches..."
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            className="min-h-[80px] resize-none"
          />
          <Button 
            onClick={handleSearch} 
            disabled={!query.trim() || isSearching}
            className="w-full"
          >
            {isSearching ? "Searching..." : "Find Similar Patches"}
          </Button>
        </div>

        {results.length > 0 && (
          <div className="space-y-3">
            <h4 className="font-medium text-sm text-muted-foreground">
              Similar Patches Found ({results.length})
            </h4>
            
            {results.map(({ patch, similarity }) => (
              <div
                key={patch.id}
                className="border rounded-lg p-3 space-y-2 bg-muted/50"
              >
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-2">
                    <span className="font-mono text-sm text-primary">{patch.id}</span>
                    <Badge
                      variant="outline"
                      className={getSimilarityColor(similarity)}
                    >
                      {Math.round(similarity * 100)}% Match
                    </Badge>
                  </div>
                  
                  {similarity > 0.9 && (
                    <div className="flex items-center gap-1 text-destructive">
                      <AlertCircle className="h-4 w-4" />
                      <span className="text-xs">High Risk</span>
                    </div>
                  )}
                </div>
                
                <p className="text-sm text-muted-foreground line-clamp-2">
                  {patch.prompt}
                </p>
                
                <div className="flex items-center gap-2 text-xs">
                  <Badge variant={patch.active ? "default" : "secondary"}>
                    {patch.active ? "Active" : "Inactive"}
                  </Badge>
                  <span className="text-muted-foreground">
                    Action: {patch.action || "deny"}
                  </span>
                  <span className="text-muted-foreground">
                    Source: {patch.source}
                  </span>
                </div>
              </div>
            ))}
          </div>
        )}

        {query && results.length === 0 && !isSearching && (
          <div className="text-center py-6 text-muted-foreground">
            <Search className="h-8 w-8 mx-auto mb-2 opacity-50" />
            <p>No similar patches found</p>
          </div>
        )}
      </CardContent>
    </Card>
  );
}