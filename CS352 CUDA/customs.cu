/*
/   Code by SM, based on customs.c from CS305
/   C++ version of the eventual CUDA implementation.
/   Simulates a day in the life of some customs agents, sequentially.
/   Change the constants to edit the parameters of the simulation.
*/
#include <stdlib.h>
#include <time.h>
#include <iostream>
#include <vector>
#include <cuda_runtime.h>
#include "device_launch_parameters.h"
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

#define MIN_CHILDREN 1 //default 1
#define MAX_CHILDREN 3 //default 3
#define CHILD_MOD1 4 //default 4
#define CHILD_MOD2 4 //default 4
#define CITIZEN_CHANCE 5 //default 5 (80%)
#define WORK_HOURS 8 //default 8
#define NORMAL_PAY 20 //default 20
#define OVERTIME_PAY 30 //default 30
#define NUM_AGENTS 20 //default 20
#define NUM_GROUPS 1000 //default 1000
#define SEED NULL //default NULL (0)

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
__global__ void calc_time(thrust::device_vector<agent*> agents, int total_time) {
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

    //
    thrust::device_vector<agent*> agents;
    thrust::fill(agents.begin(), agents.begin()+NUM_AGENTS, create_agent());
    for (int i = 0; i < NUM_AGENTS; i++) {
        thrust::fill((*agents[i]).group.begin(), (*agents[i]).group.begin()+NUM_GROUPS, create_group());
    }

    int elapsed = 0;
    calc_time<<<1, NUM_AGENTS>>>(agents, elapsed);
    int payroll = calc_payroll(elapsed);
    int average = (elapsed / NUM_AGENTS);

    std::cout << "-- Simulation Parameters --\n" << std::endl;
    std::cout << "Minimum Number of Children: " << MIN_CHILDREN << "\n" << std::endl;
    std::cout << "Maximum Number of Children: " << MAX_CHILDREN << "\n" << std::endl;
    std::cout << "Child Mod 1: " << CHILD_MOD1 << "\n" << std::endl;
    std::cout << "Child Mod 2: " << CHILD_MOD2 << "\n" << std::endl;
    std::cout << "Citizen Chance Modifier: " << CITIZEN_CHANCE << "\n" << std::endl;
    std::cout << "Work Hours: " << WORK_HOURS << "\n" << std::endl;
    std::cout << "Normal Hourly Pay: " << NORMAL_PAY << "\n" << std::endl;
    std::cout << "Overtime Hourly Pay: " << OVERTIME_PAY << "\n" << std::endl;
    std::cout << "Number of Customs Agents: " << NUM_AGENTS << "\n" << std::endl;
    std::cout << "Number of Groups per Agent: " << NUM_GROUPS << "\n" << std::endl;
    std::cout << "Random Number Seed: " << SEED << "\n" << std::endl;
    std::cout << "\n" << std::endl;
    std::cout << "-- Simulation Results --\n" << std::endl;
    std::cout << "Total Time Elapsed: " << elapsed << "\n" << std::endl;
    std::cout << "Total Payroll for the day: " << payroll << "\n" << std::endl;
    std::cout << "Average Time Elapsed: " << average << "\n" << std::endl;

    for (int i = 0; i < NUM_AGENTS; i++) {
        delete(agents[i]);
    }
}