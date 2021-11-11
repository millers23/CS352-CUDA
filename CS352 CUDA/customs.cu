/*
/   Code by SM, based on customs.c from CS305
/   C++ version of the eventual CUDA implementation.
/   Simulates a day in the life of some customs agents, sequentially.
/   Change the constants to edit the parameters of the simulation.
*/
#include <stdlib.h>
#include <time.h>
#include <vector>
#include <cuda_runtime.h>
#include "device_launch_parameters.h"
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

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

typedef struct group_struct {
    int adults;
    int children;
    bool usa;
    thrust::device_vector<group*> group;
}group;

typedef struct agent_struct {
    int timecard;
    int avail;
    thrust::device_vector<group*> group;
}agent;

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

__device__ void enqueue(agent* agt, group* grp) {
    if (agt == nullptr || grp == nullptr)
        return;

    agt->group.push_back(grp);
}

__device__ group* dequeue(agent* agt) {
    if (agt->group.front() == nullptr || agt == nullptr)
        return nullptr;
    group* grp = agt->group.back();
    agt->group.pop_back();
    return grp;
}

//this is parallel now
__global__ void calc_time(std::vector<agent*> agents, int total_time) {
    int max_time = 0;

    int i = threadIdx.x;
    agent* a = agents[i];
    int agent_time = 0;
    group* g = dequeue(a);

    while (g != nullptr) {
        int temp = g->adults;
        if (!g->usa)
            temp *= 2;
        temp += (1 + g->children) / 2;
        agent_time += temp;
        delete(g);
        g = dequeue(a);
    }

    if (agent_time > max_time)
        max_time = agent_time;
    if (max_time > total_time)
        total_time = max_time;
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
int main() {
    srand(time(SEED));

    thrust::device_vector<agent*> agents;
    thrust::fill(agents.begin(), agents.begin()+NUM_AGENTS, create_agent());
    for (int i = 0; i < NUM_AGENTS; i++) {
        thrust::fill((*agents[i]).group.begin(), (*agents[i]).group.begin()+NUM_GROUPS, create_group());
    }

    int elapsed = 0;
    calc_time<<<1, NUM_AGENTS>>>(agents, elapsed);
    int payroll = calc_payroll(elapsed);
    int average = (elapsed / NUM_AGENTS);

    for (int i = 0; i < NUM_AGENTS; i++) {
        delete(agents[i]);
    }
}