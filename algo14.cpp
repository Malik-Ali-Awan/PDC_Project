/**
 * SCC/CAC Partitioning Algorithms Implementation
 * Algorithms 1-4 from the paper
 * 
 * This implementation uses MPI and OpenMP for parallelization
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
 
 // Vertex structure
 struct Vertex {
     int id;
     int index;
     int lowlink;
     int level;
     int depth;
     int type;  // 0: undefined, 1: scc, 2: cac
     bool onStack;
     std::vector<int> neighbors;
     std::vector<double> weights;  // For weighted edges
 };
 
 // Graph structure
 class Graph {
 private:
     std::unordered_map<int, Vertex> vertices;
     std::vector<std::pair<int, int>> edges;
     std::vector<std::pair<int, int>> weighted_edges;
     std::unordered_map<int, std::vector<int>> components;
     std::unordered_map<int, int> componentTypes;  // 1: SCC, 2: CAC
     int index;
 
 public:
     Graph() : index(0) {}
 
     // Add vertex
     void addVertex(int id) {
         if (vertices.find(id) == vertices.end()) {
             Vertex v;
             v.id = id;
             v.index = -1;
             v.lowlink = -1;
             v.level = 0;
             v.depth = 0;
             v.type = 0;  // undefined
             v.onStack = false;
             vertices[id] = v;
         }
     }
 
     // Add edge
     void addEdge(int from, int to, double weight = 1.0) {
         // Add vertices if they don't exist
         addVertex(from);
         addVertex(to);
 
         // Add neighbors and edge
         vertices[from].neighbors.push_back(to);
         vertices[from].weights.push_back(weight);
         edges.push_back({from, to});
         if (weight != 1.0) {
             weighted_edges.push_back({from, to});
         }
     }
 
     // Get vertices
     std::unordered_map<int, Vertex>& getVertices() {
         return vertices;
     }
 
     // Get edges
     std::vector<std::pair<int, int>>& getEdges() {
         return edges;
     }
 
     // Get components
     std::unordered_map<int, std::vector<int>>& getComponents() {
         return components;
     }
 
     // Get component types
     std::unordered_map<int, int>& getComponentTypes() {
         return componentTypes;
     }
 
     // Execute SCC/CAC partitioning algorithm (Algorithm 1)
     void partitionSCCCAC() {
         // Initialize
         index = 0;
         
         // For each vertex in the graph
         for (auto& v_pair : vertices) {
             // If vertex not yet discovered
             if (v_pair.second.index == -1) {
                 // Algorithm 1: SCC/CAC partitioning
                 discover(v_pair.first);
             }
         }
     }
 
     // Algorithm 2: Discover function
    void discover(int v_id) {
        Vertex& v = vertices[v_id];
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
            Vertex& current = vertices[current_id];
            bool should_pop = true;

            for (size_t i = 0; i < current.neighbors.size(); i++) {
                int w_id = current.neighbors[i];
                Vertex& w = vertices[w_id];

                if (w.index == -1) {
                    w.index = index;
                    w.lowlink = index;
                    w.level = 1;
                    w.depth = 1;
                    index++;
                    dfs_stack.push(w_id);
                    w.onStack = true;
                    should_pop = false;
                    break; // Process this new node first
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
 
     // Algorithm 3: Explore function
     void explore(int v_id, int w_id) {
         Vertex& v = vertices[v_id];
         Vertex& w = vertices[w_id];
 
         // If w is not initialized (not discovered)
         if (w.index == -1) {
             // DFS(w) // Discover and Explore
             discover(w_id);
             finish(w_id);
         }
 
         // At this point w is either in a component already
         // or belongs to the same SCC as v
         if (w.type != 0) {  // If w component type is defined
             v.level = std::max(v.level, w.level + 1);
         } else {
             // w belongs to the same SCC
             v.level = std::max(v.level, w.level);
             v.lowlink = std::min(v.lowlink, w.lowlink);
         }
     }
 
     // Algorithm 4: Finish function
    void finish(int v_id) {
        Vertex& v = vertices[v_id];

        if (v.lowlink == v.index) {
            std::vector<int> component;
            std::stack<int> component_stack;
            component_stack.push(v_id);

            while (!component_stack.empty()) {
                int w_id = component_stack.top();
                Vertex& w = vertices[w_id];

                if (w.lowlink == v.index) {
                    component_stack.pop();
                    component.push_back(w_id);
                    w.type = 1; // SCC
                    w.level = v.level;
                    
                    for (int neighbor_id : w.neighbors) {
                        Vertex& neighbor = vertices[neighbor_id];
                        if (neighbor.onStack && neighbor.lowlink == v.index) {
                            component_stack.push(neighbor_id);
                        }
                    }
                }
            }

            components[v.index] = component;
            componentTypes[v.index] = 1; // SCC

            if (component.size() == 1) {
                v.type = 2; // CAC
                componentTypes[v.index] = 2;
                
                // Simplified merge detection
                bool has_merge = false;
                for (int neighbor_id : v.neighbors) {
                    Vertex& neighbor = vertices[neighbor_id];
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
 
     // Find a component for a vertex
     int findComponentForVertex(int v_id) {
         // Find which component contains the vertex
         for (const auto& comp_pair : components) {
             for (int vertex : comp_pair.second) {
                 if (vertex == v_id) {
                     return comp_pair.first;
                 }
             }
         }
         return -1;
     }
 
     // Print graph statistics
     void printStats() {
         std::cout << "Graph Statistics:" << std::endl;
         std::cout << "Number of vertices: " << vertices.size() << std::endl;
         std::cout << "Number of edges: " << edges.size() << std::endl;
         std::cout << "Number of components: " << components.size() << std::endl;
         
         // Count components by type
         int scc_count = 0, cac_count = 0;
         for (const auto& type_pair : componentTypes) {
             if (type_pair.second == 1) scc_count++;
             else if (type_pair.second == 2) cac_count++;
         }
         
         std::cout << "Number of SCC components: " << scc_count << std::endl;
         std::cout << "Number of CAC components: " << cac_count << std::endl;
     }
 };
 
 // Function to read edge list from gzipped file
 void readEdgeList(const std::string& filename, Graph& graph) {
     gzFile file = gzopen(filename.c_str(), "r");
     if (!file) {
         std::cerr << "Failed to open file: " << filename << std::endl;
         return;
     }
 
     char buffer[1024];
     std::string line;
     
     while (gzgets(file, buffer, sizeof(buffer)) != NULL) {
         line = buffer;
         std::istringstream iss(line);
         int from, to;
         double weight = 1.0;
         
         if (iss >> from >> to) {
             // Check if there's a weight
             if (iss >> weight) {
                 graph.addEdge(from, to, weight);
             } else {
                 graph.addEdge(from, to);
             }
         }
     }
     
     gzclose(file);
 }
 
 // Parallel implementation of graph partitioning
 void parallelGraphPartitioning(Graph& graph, int world_rank, int world_size) {
     // Get all vertices
     auto& vertices = graph.getVertices();
     std::vector<int> vertex_ids;
     
     for (const auto& v_pair : vertices) {
         vertex_ids.push_back(v_pair.first);
     }
     
     // Divide vertices among MPI processes
     int vertices_per_process = vertex_ids.size() / world_size;
     int start_idx = world_rank * vertices_per_process;
     int end_idx = (world_rank == world_size - 1) ? vertex_ids.size() : (world_rank + 1) * vertices_per_process;
     
     // Process local vertices
     #pragma omp parallel for
     for (int i = start_idx; i < end_idx; i++) {
         int v_id = vertex_ids[i];
         if (vertices[v_id].index == -1) {
             graph.discover(v_id);
         }
     }
     
     // MPI synchronization of partitioning data would be done here
     // This is a simplified version
     MPI_Barrier(MPI_COMM_WORLD);
 }
 
int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    
    int world_rank, world_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    
    omp_set_num_threads(4);
    
    // All ranks participate
    Graph graph;
    std::string filename = "higgs-social_network.edgelist.gz";
    
    // Rank 0 reads and broadcasts the graph
    if (world_rank == 0) {
        readEdgeList(filename, graph);
    }
    
    // Simple parallelization - in practice you'd need proper graph distribution
    parallelGraphPartitioning(graph, world_rank, world_size);
    
    if (world_rank == 0) {
        std::cout << "After partitioning: ";
        graph.printStats();
    }
    
    MPI_Finalize();
    return 0;
}
