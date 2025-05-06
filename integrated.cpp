/**
 * Integrated Social Network Analysis System
 * Combines:
 * - SCC/CAC Partitioning (Algorithm 14)
 * - Influence Power Measurement (Algorithm 5)
 * 
 * Uses MPI and OpenMP for parallelization
 */

#include <iostream>
#include <fstream>
#include <vector>
#include <stack>
#include <unordered_map>
#include <unordered_set>
#include <mpi.h>
#include <omp.h>
#include <zlib.h>
#include <string>
#include <sstream>
#include <algorithm>
#include <limits>
#include <cstring>
#include <cmath>
#include <cstdlib>
#include <ctime>

// ======================== SCC/CAC Partitioning (Algorithm 14) ========================

struct SCC_Vertex {
    int id;
    int index;
    int lowlink;
    int level;
    int depth;
    int type;  // 0: undefined, 1: scc, 2: cac
    bool onStack;
    double influence_power;  // Added for integration
    std::vector<int> neighbors;
    std::vector<double> weights;
};

class SCC_Graph {
private:
    std::unordered_map<int, SCC_Vertex> vertices;
    std::vector<std::pair<int, int>> edges;
    std::unordered_map<int, std::vector<int>> components;
    std::unordered_map<int, int> componentTypes;
    int index;

public:
    SCC_Graph() : index(0) {}

    void addVertex(int id) {
        if (vertices.find(id) == vertices.end()) {
            SCC_Vertex v;
            v.id = id;
            v.index = -1;
            v.lowlink = -1;
            v.level = 0;
            v.depth = 0;
            v.type = 0;
            v.onStack = false;
            v.influence_power = 0.0;
            vertices[id] = v;
        }
    }

    void addEdge(int from, int to, double weight = 1.0) {
        addVertex(from);
        addVertex(to);
        vertices[from].neighbors.push_back(to);
        vertices[from].weights.push_back(weight);
        edges.push_back({from, to});
    }

    void setVertexInfluence(int id, double influence) {
        if (vertices.find(id) != vertices.end()) {
            vertices[id].influence_power = influence;
        }
    }

    void partitionSCCCAC() {
        index = 0;
        for (auto& v_pair : vertices) {
            if (v_pair.second.index == -1) {
                discover(v_pair.first);
            }
        }
    }

    void discover(int v_id) {
        SCC_Vertex& v = vertices[v_id];
        v.index = index;
        v.lowlink = index;
        v.level = 1;
        v.depth = 1;
        index++;

        std::stack<int> dfs_stack;
        dfs_stack.push(v_id);
        v.onStack = true;

        while (!dfs_stack.empty()) {
            int current_id = dfs_stack.top();
            SCC_Vertex& current = vertices[current_id];
            bool should_pop = true;

            for (size_t i = 0; i < current.neighbors.size(); i++) {
                int w_id = current.neighbors[i];
                SCC_Vertex& w = vertices[w_id];

                if (w.index == -1) {
                    w.index = index;
                    w.lowlink = index;
                    w.level = 1;
                    w.depth = 1;
                    index++;
                    dfs_stack.push(w_id);
                    w.onStack = true;
                    should_pop = false;
                    break;
                }
                else {
                    if (w.onStack) {
                        current.lowlink = std::min(current.lowlink, w.index);
                    }
                    if (w.type != 0) {
                        current.level = std::max(current.level, w.level + 1);
                    }
                }
            }

            if (should_pop) {
                dfs_stack.pop();
                current.onStack = false;
                finish(current_id);
            }
        }
    }

    void finish(int v_id) {
        SCC_Vertex& v = vertices[v_id];

        if (v.lowlink == v.index) {
            std::vector<int> component;
            std::stack<int> component_stack;
            component_stack.push(v_id);

            while (!component_stack.empty()) {
                int w_id = component_stack.top();
                SCC_Vertex& w = vertices[w_id];

                if (w.lowlink == v.index) {
                    component_stack.pop();
                    component.push_back(w_id);
                    w.type = 1;
                    w.level = v.level;
                    
                    for (int neighbor_id : w.neighbors) {
                        SCC_Vertex& neighbor = vertices[neighbor_id];
                        if (neighbor.onStack && neighbor.lowlink == v.index) {
                            component_stack.push(neighbor_id);
                        }
                    }
                }
            }

            components[v.index] = component;
            componentTypes[v.index] = 1;

            if (component.size() == 1) {
                v.type = 2;
                componentTypes[v.index] = 2;
                
                bool has_merge = false;
                for (int neighbor_id : v.neighbors) {
                    SCC_Vertex& neighbor = vertices[neighbor_id];
                    if ((neighbor.type == 1 || neighbor.type == 2) && 
                        neighbor.level == v.level - 1) {
                        has_merge = true;
                        break;
                    }
                }
                
                if (has_merge) {
                    v.level = v.level - 1;
                }
            }
        }
    }

    std::unordered_map<int, SCC_Vertex>& getVertices() { return vertices; }
    std::vector<std::pair<int, int>>& getEdges() { return edges; }
    std::unordered_map<int, std::vector<int>>& getComponents() { return components; }

    void printStats() {
        std::cout << "SCC/CAC Graph Statistics:" << std::endl;
        std::cout << "Vertices: " << vertices.size() << ", Edges: " << edges.size() << std::endl;
        
        int scc_count = 0, cac_count = 0;
        for (const auto& type_pair : componentTypes) {
            if (type_pair.second == 1) scc_count++;
            else if (type_pair.second == 2) cac_count++;
        }
        
        std::cout << "SCC components: " << scc_count << ", CAC components: " << cac_count << std::endl;
    }
};

// ======================== Influence Power Measurement (Algorithm 5) ========================

struct InfluenceVertex {
    int id;
    double influence_power;
    std::vector<int> followers;
    std::vector<int> following;
    std::vector<double> edge_weights;
    int interest_category;
};

class InfluenceGraph {
private:
    std::unordered_map<int, InfluenceVertex> vertices;
    std::vector<std::pair<int, int>> edges;
    std::vector<int> interest_categories;
    std::vector<double> friendship_factors;

    struct pair_hash {
        template <class T1, class T2>
        std::size_t operator () (const std::pair<T1, T2> &p) const {
            auto h1 = std::hash<T1>{}(p.first);
            auto h2 = std::hash<T2>{}(p.second);
            return h1 ^ h2;
        }
    };

    std::unordered_map<std::pair<int, int>, double, pair_hash> weights;

public:
    const std::unordered_map<int, InfluenceVertex>& getVertices() const {
        return vertices;
    }
    
    double getVertexInfluence(int id) const {
        auto it = vertices.find(id);
        if (it != vertices.end()) {
            return it->second.influence_power;
        }
        return 0.0;
    }
    InfluenceGraph() {
        interest_categories = {1, 2, 3, 4, 5};
        friendship_factors = {0.1, 0.2, 0.3, 0.4, 0.5};
    }

    void addVertex(int id) {
        if (vertices.find(id) == vertices.end()) {
            InfluenceVertex v;
            v.id = id;
            v.influence_power = 0.0;
            v.interest_category = interest_categories[rand() % interest_categories.size()];
            vertices[id] = v;
        }
    }

    void addEdge(int from, int to, double weight = 1.0) {
        addVertex(from);
        addVertex(to);
        vertices[from].following.push_back(to);
        vertices[to].followers.push_back(from);
        weights[{from, to}] = weight;
        edges.push_back({from, to});
    }

    double getEdgeWeight(int from, int to) {
        auto it = weights.find({from, to});
        return (it != weights.end()) ? it->second : 0.0;
    }

    double calculateCI(int u_id, int v_id) {
        InfluenceVertex& u = vertices[u_id];
        InfluenceVertex& v = vertices[v_id];
        return (u.interest_category == v.interest_category) ? 1.0 : 0.5;
    }

    int calculateMutualFriends(int u_id, int v_id) {
        InfluenceVertex& u = vertices[u_id];
        InfluenceVertex& v = vertices[v_id];
        std::unordered_set<int> u_followers(u.followers.begin(), u.followers.end());
        
        int mutual_count = 0;
        for (int follower : v.followers) {
            if (u_followers.find(follower) != u_followers.end()) {
                mutual_count++;
            }
        }
        return mutual_count;
    }

    void calculateInfluencePower(int num_cpus, double d = 0.85) {
        int n = 0;
        int max_level = 3;
        int num_components = vertices.size() / 10;
        if (num_components < 1) num_components = 1;
        
        while (n < max_level) {
            #pragma omp parallel for num_threads(num_cpus)
            for (int p = 1; p <= num_cpus; p++) {
                int c = (num_components / num_cpus) + ((p - 1) * num_components / num_cpus);
                
                #pragma omp parallel for
                for (int i = c; i < c + (num_components / num_cpus); i++) {
                    int start_idx = i * 10;
                    int end_idx = std::min((i + 1) * 10, (int)vertices.size());
                    
                    std::vector<int> vertex_ids;
                    for (const auto& v_pair : vertices) {
                        vertex_ids.push_back(v_pair.first);
                    }
                    
                    for (int idx = start_idx; idx < end_idx && idx < vertex_ids.size(); idx++) {
                        int u_id = vertex_ids[idx];
                        InfluenceVertex& u = vertices[u_id];
                        
                        for (int v_id : u.followers) {
                            InfluenceVertex& v = vertices[v_id];
                            double ci = calculateCI(u_id, v_id);
                            double na = calculateMutualFriends(u_id, v_id);
                            double alpha = friendship_factors[n % friendship_factors.size()];
                            double psi = alpha * ci * na;
                            
                            #pragma omp atomic
                            v.influence_power += psi;
                        }
                    }
                }
            }
            
            for (auto& v_pair : vertices) {
                InfluenceVertex& v = v_pair.second;
                double f_ui = (double)v.followers.size() / vertices.size();
                
                double sum = 0.0;
                for (int follower_id : v.followers) {
                    double weight = getEdgeWeight(follower_id, v.id);
                    double follower_followers = vertices[follower_id].followers.size();
                    if (follower_followers == 0) follower_followers = 1;
                    sum += weight * f_ui / follower_followers;
                }
                
                v.influence_power = (1 - d) * f_ui + d * sum;
            }
            
            n++;
        }
    }

    std::vector<int> getTopKInfluential(int k) {
        std::vector<std::pair<int, double>> influence_pairs;
        for (const auto& v_pair : vertices) {
            influence_pairs.push_back({v_pair.first, v_pair.second.influence_power});
        }
        
        std::sort(influence_pairs.begin(), influence_pairs.end(), 
                  [](const auto& a, const auto& b) { return a.second > b.second; });
        
        std::vector<int> top_k;
        for (int i = 0; i < k && i < influence_pairs.size(); i++) {
            top_k.push_back(influence_pairs[i].first);
        }
        return top_k;
    }

    void printStats() {
        std::cout << "Influence Graph Statistics:" << std::endl;
        std::cout << "Vertices: " << vertices.size() << ", Edges: " << edges.size() << std::endl;
        
        double total_influence = 0.0;
        for (const auto& v_pair : vertices) {
            total_influence += v_pair.second.influence_power;
        }
        
        std::cout << "Avg Influence: " << (vertices.size() > 0 ? total_influence / vertices.size() : 0.0) << std::endl;
        
        std::vector<int> top_5 = getTopKInfluential(5);
        std::cout << "Top 5: ";
        for (int user_id : top_5) {
            std::cout << user_id << "(" << vertices[user_id].influence_power << ") ";
        }
        std::cout << std::endl;
    }
};

// ======================== Common Functions ========================

void readEdgeList(const std::string& filename, SCC_Graph& graph) {
    gzFile file = gzopen(filename.c_str(), "r");
    if (!file) {
        std::cerr << "Failed to open file: " << filename << std::endl;
        return;
    }

    char buffer[1024];
    while (gzgets(file, buffer, sizeof(buffer)) != NULL) {
        std::istringstream iss(buffer);
        int from, to;
        double weight = 1.0;
        if (iss >> from >> to) {
            if (iss >> weight) graph.addEdge(from, to, weight);
            else graph.addEdge(from, to);
        }
    }
    gzclose(file);
}

void transferGraphData(SCC_Graph& sccGraph, InfluenceGraph& infGraph) {
    for (auto& v_pair : sccGraph.getVertices()) {
        int id = v_pair.first;
        infGraph.addVertex(id);
        
        for (size_t i = 0; i < v_pair.second.neighbors.size(); i++) {
            int neighbor = v_pair.second.neighbors[i];
            double weight = v_pair.second.weights[i];
            infGraph.addEdge(id, neighbor, weight);
        }
    }
}

// ======================== Main Program ========================

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int world_rank, world_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    srand(time(NULL) + world_rank);
    omp_set_num_threads(4);

    if (world_rank == 0) {
        // Phase 1: SCC/CAC Partitioning
        std::cout << "=== Phase 1: SCC/CAC Partitioning ===" << std::endl;
        SCC_Graph sccGraph;
        readEdgeList("higgs-social_network.edgelist.gz", sccGraph);
        sccGraph.partitionSCCCAC();
        sccGraph.printStats();

        // Phase 2: Influence Power Measurement
        std::cout << "\n=== Phase 2: Influence Analysis ===" << std::endl;
        InfluenceGraph infGraph;
        transferGraphData(sccGraph, infGraph);
        
        infGraph.calculateInfluencePower(omp_get_max_threads());
        infGraph.printStats();

        // Show top influencers - using the new accessor method
        std::vector<int> top_influencers = infGraph.getTopKInfluential(10);
        std::cout << "\nTop 10 Influencers:" << std::endl;
        for (int i = 0; i < top_influencers.size(); i++) {
            std::cout << (i+1) << ". User " << top_influencers[i] 
                      << " (IP: " << infGraph.getVertexInfluence(top_influencers[i]) << ")" << std::endl;
        }
    }

    MPI_Finalize();
    return 0;
}
