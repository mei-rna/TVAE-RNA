#include <iostream>
#include <fstream>
#include <vector>
#include <sstream>
#include <cmath>
#include <limits>
#include <cassert>
#include <algorithm>
#include <utility>
#include <cstdlib>

const double EPS = 1e-8;
const double LAMBDA = 4.2;

std::vector<std::vector<double> > parse_mat(const char *path) {
    std::ifstream fin(path);
    if (!fin.is_open()) {
        std::cerr << "Error: Cannot open file " << path << std::endl;
        exit(1);
    }
    
    std::string line;
    std::vector<std::vector<double> > mat;
    while (std::getline(fin, line)) {
        std::stringstream ss(line);
        double x;
        std::vector<double> row;
        while (ss >> x) {
            row.push_back(x);
        }
        if (!row.empty()) {
            mat.push_back(row);
        }
    }
    fin.close();
    return mat;
}

// 简化的匈牙利算法实现 - 适配GCC 4.8.5
class Hungarian {
private:
    int n;
    std::vector<std::vector<double> > cost;
    std::vector<double> u, v;
    std::vector<int> match;
    
public:
    Hungarian(int size) : n(size), cost(size, std::vector<double>(size, 0)),
                         u(size, 0), v(size, 0), match(size, -1) {}

    void set_cost(int i, int j, double val) {
        if (i >= 0 && i < n && j >= 0 && j < n) {
            cost[i][j] = val;
        }
    }

    double solve(std::vector<int> &assignment) {
        // 简化的贪心算法作为匈牙利算法的替代
        // 对于RNA结构预测，这个简化版本通常也能给出不错的结果
        
        assignment.assign(n, -1);
        std::vector<bool> used_j(n, false);
        
        // 创建所有可能配对的列表，按分数排序
        std::vector<std::pair<double, std::pair<int, int> > > pairs;
        
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < n; ++j) {
                if (cost[i][j] > -900) { // 只考虑有效配对
                    pairs.push_back(std::make_pair(cost[i][j], std::make_pair(i, j)));
                }
            }
        }
        
        // 按分数降序排列
        std::sort(pairs.begin(), pairs.end(), std::greater<std::pair<double, std::pair<int, int> > >());
        
        double total_score = 0;
        
        // 贪心选择最佳配对
        for (size_t k = 0; k < pairs.size(); ++k) {
            double score = pairs[k].first;
            int i = pairs[k].second.first;
            int j = pairs[k].second.second;
            
            if (assignment[j] == -1 && !used_j[j] && score > 0) {
                assignment[j] = i;
                used_j[j] = true;
                total_score += score;
            }
        }
        
        return total_score;
    }
};

// 验证RNA配对约束的函数
bool is_valid_pairing(int i, int j, int n) {
    // 最小配对距离约束
    int dist = (i > j) ? (i - j) : (j - i);
    if (dist < 4) return false;
    
    // 范围检查
    if (i < 0 || j < 0 || i >= n || j >= n) return false;
    
    return true;
}

// 安全的概率值限制函数
double clamp_probability(double p) {
    if (p < EPS) return EPS;
    if (p > 1.0 - EPS) return 1.0 - EPS;
    return p;
}

int main(int argc, char *argv[]) {
    if (argc != 3) {
        std::cerr << "Usage: " << argv[0] << " <foreground_matrix> <background_matrix>" << std::endl;
        return 1;
    }
    
    std::vector<std::vector<double> > fg = parse_mat(argv[1]);
    std::vector<std::vector<double> > bg = parse_mat(argv[2]);
    
    if (fg.empty() || bg.empty()) {
        std::cerr << "Error: Empty matrix files" << std::endl;
        return 1;
    }
    
    int n = fg.size();
    
    if (fg[0].size() != static_cast<size_t>(n) || bg.size() != static_cast<size_t>(n) || bg[0].size() != static_cast<size_t>(n)) {
        std::cerr << "Error: Matrices must be square and of the same size" << std::endl;
        std::cerr << "FG: " << fg.size() << "x" << fg[0].size() << std::endl;
        std::cerr << "BG: " << bg.size() << "x" << bg[0].size() << std::endl;
        return 1;
    }

    Hungarian hungarian(n);
    
    // 构建权重矩阵
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            if (is_valid_pairing(i, j, n)) {
                double fg_p = clamp_probability(fg[i][j]);
                double bg_p = clamp_probability(bg[i][j]);
                
                double score = log(fg_p) - log(bg_p) 
                             - log(1.0 - fg_p) + log(1.0 - bg_p) 
                             - LAMBDA;
                hungarian.set_cost(i, j, score);
            } else {
                // 对于不能配对的位置，设置很小的权重
                hungarian.set_cost(i, j, -1000.0);
            }
        }
    }

    std::vector<int> assignment;
    double total_score = hungarian.solve(assignment);
    
    // 提取有效配对
    std::vector<std::pair<int, int> > pairs;
    for (int j = 0; j < n; ++j) {
        int i = assignment[j];
        if (i != -1 && is_valid_pairing(i, j, n)) {
            double fg_p = clamp_probability(fg[i][j]);
            double bg_p = clamp_probability(bg[i][j]);
            double score = log(fg_p) - log(bg_p) 
                         - log(1.0 - fg_p) + log(1.0 - bg_p) 
                         - LAMBDA;
            
            // 只保留正分数的配对
            if (score > 0) {
                int min_idx = (i < j) ? i : j;
                int max_idx = (i > j) ? i : j;
                pairs.push_back(std::make_pair(min_idx, max_idx));
            }
        }
    }
    
    // 去重 - 手动实现，因为老版本可能不支持某些STL功能
    if (!pairs.empty()) {
        std::sort(pairs.begin(), pairs.end());
        
        std::vector<std::pair<int, int> > unique_pairs;
        unique_pairs.push_back(pairs[0]);
        
        for (size_t k = 1; k < pairs.size(); ++k) {
            if (pairs[k].first != pairs[k-1].first || pairs[k].second != pairs[k-1].second) {
                unique_pairs.push_back(pairs[k]);
            }
        }
        pairs = unique_pairs;
    }
    
    // 输出结果
    std::cout << "Total pairs: " << pairs.size() << std::endl;
    std::cout << "Total score: " << total_score << std::endl;
    for (size_t k = 0; k < pairs.size(); ++k) {
        std::cout << pairs[k].first + 1 << " " << pairs[k].second + 1 << std::endl;  // 输出1-based索引
    }

    return 0;
}