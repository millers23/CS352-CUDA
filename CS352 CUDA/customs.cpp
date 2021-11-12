/*
/   Code by SM, based on customs.c from CS305
/   C++ version of the eventual CUDA implementation.
/   Simulates a day in the life of some customs agents, sequentially.
/   Change the constants to edit the parameters of the simulation.
*/
#include <stdlib.h>
#include <time.h>
#include <vector>

#define MIN_CHILDREN 1
#define MAX_CHILDREN 3
#define CHILD_MOD1 4
#define CHILD_MOD2 4
#define CITIZEN_CHANCE 5
#define WORK_HOURS 8
#define NORMAL_PAY 20
#define OVERTIME_PAY 30
#define NUM_AGENTS 20
#define NUM_GROUPS 1000
#define SEED NULL

struct group {
    int adults;
    int children;
    bool usa;
    group(int a, int c, bool u) 
        : adults(a), children(c), usa(u)
    {}
};

struct agent {
    int timecard;
    int avail;
    std::vector<group*> group;//rename
    agent(int t, int a, std::vector<group*> g) 
        : timecard(t), avail(a), group(g)
    {}
};

typedef struct stats_struct {
    int total_time;
    int total_payroll;
    int avg_wait_time;
    int max_wait_time;
}stats;

group* create_group() {
    group* g = new group;
    g->adults = MIN_CHILDREN + rand() % MAX_CHILDREN;
    g->children = (rand() % CHILD_MOD1) + (rand() % CHILD_MOD2) - 2;
    if (g->children < 0)
        g->children = 0;
    g->usa = ((rand() % CITIZEN_CHANCE) == 0) ? false : true;
    return g;
}

agent* create_agent() {
    agent* a = new agent;
    a->timecard = 0;
    a->avail = 0;
    return a;
}

int proc_time(group* grp) {
    int result = grp->adults;
    if (!grp->usa)
        result *= 2;
    result += (1 + grp->children) / 2;
    return result;
}

void enqueue(agent* agt, group* grp) {
    if (agt == nullptr || grp == nullptr)
        return;

    agt->group.push_back(grp);
}

group* dequeue(agent* agt) {
    if (agt->group.front() == nullptr || agt == nullptr)
        return nullptr;
    group *grp = agt->group.back();
    agt->group.pop_back();
    return grp;
}

//this will be parallel in CUDA
int calc_time(std::vector<agent*> agents) {
    int max_time = 0;
    int total_time = 0;

    for (int i = 0; i < NUM_AGENTS; i++) {
        agent* a = agents[i];
        int agent_time = 0;
        group* g = dequeue(a);

        while (g != nullptr) {
            agent_time += proc_time(g);
            delete(g);
            g = dequeue(a);
        }

        if (agent_time > max_time)
            max_time = agent_time;
        if (max_time > total_time)
            total_time = max_time;
    }
    return total_time;
}

int calc_payroll(int time) {
    int cost = 0;
    int overtime = 0;

    if ((time / 60) <= WORK_HOURS)
        cost = (time / 60) * NORMAL_PAY * NUM_AGENTS;
    if ((time / 60) > WORK_HOURS) {
        cost = WORK_HOURS * NORMAL_PAY * NUM_AGENTS;
        cost += ((time / 60) - 8) * OVERTIME_PAY * NUM_AGENTS;
    }
    return cost;
}

//let's do this!
int main(int argc, char* argv[]) {
    srand(time(SEED));

    std::vector<agent*> agents;
    for (int i = 0; i < NUM_AGENTS; i++) {
        agents.push_back(create_agent());
    }
    for (int i = 0; i < NUM_GROUPS; i++) {
        enqueue(agents[i], create_group());
    }

    int elapsed = calc_time(agents);
    int payroll = calc_payroll(elapsed);
    int average = (elapsed / NUM_AGENTS);

    for (int i = 0; i < NUM_AGENTS; i++) {
        delete(agents[i]);
    }
}