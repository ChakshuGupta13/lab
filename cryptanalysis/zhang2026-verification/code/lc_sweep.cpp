// Parameter sweep: for a given R, scan all (V, K) pairs and report
// the best local collisions found, sorted by OBJ.
// Usage: ./scratch_sweep [R] [max_obj]
//   R defaults to 37, max_obj defaults to 15.

#include "local_collision.hpp"
#include <cstdio>
#include <cstdlib>
#include <algorithm>
#include <set>

using namespace local_collision;

int main(int argc, char* argv[]) {
    int R = (argc > 1) ? atoi(argv[1]) : 37;
    int max_obj = (argc > 2) ? atoi(argv[2]) : 15;

    printf("Sweep: R=%d, max_obj=%d\n", R, max_obj);
    printf("Scanning all (V, K) with K>=2, V>=0, V+K<=R ...\n\n");

    // Collect all results across all (V, K)
    std::vector<LocalCollision> all;

    for (int V = 0; V < R; ++V) {
        for (int K = 2; V + K <= R; ++K) {
            auto results = search_local_collisions(R, V, K, max_obj);
            for (auto& lc : results)
                all.push_back(std::move(lc));
        }
    }

    // Sort by OBJ, then by number of active words, then by V
    std::sort(all.begin(), all.end(), [](const LocalCollision& a, const LocalCollision& b) {
        if (a.obj != b.obj) return a.obj < b.obj;
        if (a.active_words.size() != b.active_words.size())
            return a.active_words.size() < b.active_words.size();
        return a.V < b.V;
    });

    // Deduplicate by active word set (same LC can't appear from different (V,K)
    // but just in case)
    std::set<std::vector<int>> seen;
    std::vector<LocalCollision> unique;
    for (auto& lc : all) {
        if (seen.insert(lc.active_words).second)
            unique.push_back(std::move(lc));
    }

    printf("Found %d unique local collisions (OBJ <= %d)\n\n", (int)unique.size(), max_obj);
    printf("%-5s %-5s %-5s %-5s %-6s  %-40s  %s\n",
           "OBJ", "V", "K", "#act", "#canc", "Active words", "Cancel steps");
    printf("%-5s %-5s %-5s %-5s %-6s  %-40s  %s\n",
           "---", "---", "---", "---", "-----", "-------------", "------------");

    for (auto& lc : unique) {
        char act_buf[256] = {0}, canc_buf[256] = {0};
        int pos = 0;
        for (int w : lc.active_words)
            pos += snprintf(act_buf + pos, sizeof(act_buf) - pos, "W%d ", w);
        pos = 0;
        for (int s : lc.cancel_steps)
            pos += snprintf(canc_buf + pos, sizeof(canc_buf) - pos, "%d ", s);

        printf("%-5d %-5d %-5d %-5d %-6d  %-40s  %s\n",
               lc.obj, lc.V, lc.K, (int)lc.active_words.size(),
               (int)lc.cancel_steps.size(), act_buf, canc_buf);
    }

    return 0;
}
