#include <iostream>
#include <fstream>
#include <sstream>
#include <unordered_map>
#include <vector>
#include <algorithm>
#include <string>

struct Edge {
    int target;
    int retweet;
    int reply;
    int mention;
};

typedef std::unordered_map<int, std::vector<Edge>> Graph;
typedef std::unordered_map<int, double> InfluenceMap;

Graph read_graph(const std::string& filename) {
    Graph graph;
    std::ifstream infile(filename);
    if (!infile.is_open()) {
        std::cerr << "Error opening file: " << filename << std::endl;
        return graph;
    }

    int source, target, retweet, reply, mention;
    while (infile >> source >> target >> retweet >> reply >> mention) {
        graph[source].push_back({target, retweet, reply, mention});
    }

    return graph;
}

InfluenceMap calculate_influence(const Graph& graph, int steps = 3) {
    InfluenceMap influence;

    // Initialize influence score
    for (Graph::const_iterator it = graph.begin(); it != graph.end(); ++it) {
        influence[it->first] = 1.0;
    }

    for (int step = 0; step < steps; ++step) {
        InfluenceMap new_influence = influence;

        for (Graph::const_iterator it = graph.begin(); it != graph.end(); ++it) {
            int src = it->first;
            const std::vector<Edge>& edges = it->second;

            for (std::vector<Edge>::const_iterator eit = edges.begin(); eit != edges.end(); ++eit) {
                double weight = eit->retweet * 0.5 + eit->reply * 0.3 + eit->mention * 0.2;
                new_influence[eit->target] += influence[src] * weight;
            }
        }

        influence = new_influence;
    }

    return influence;
}

void print_top_influencers(const InfluenceMap& influence, int top_n = 10) {
    std::vector<std::pair<int, double>> sorted(influence.begin(), influence.end());

    std::sort(sorted.begin(), sorted.end(), 
        [](const std::pair<int, double>& a, const std::pair<int, double>& b) {
            return b.second < a.second;
        });

    std::cout << "\nTop " << top_n << " Influential Users:\n";
    for (int i = 0; i < std::min(top_n, (int)sorted.size()); ++i) {
        std::cout << "User " << sorted[i].first << " → Influence Score: " << sorted[i].second << "\n";
    }
}

int main() {
    std::string filename = "combined_higgs_dataset.edgelist";
    Graph graph = read_graph(filename);

    if (graph.empty()) {
        std::cerr << "Graph is empty or file not found.\n";
        return 1;
    }

    std::cout << "Graph loaded with " << graph.size() << " users.\n";

    InfluenceMap influence = calculate_influence(graph);

    print_top_influencers(influence);

    return 0;
}
