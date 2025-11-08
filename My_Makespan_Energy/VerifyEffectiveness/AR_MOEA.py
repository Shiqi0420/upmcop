import random, copy, math, os, sys

sys.path.append("./")
from Tool import myRandom
import numpy as np


class ARMOEA:
    def __init__(self, popSize, maxGen, T, taskNumberRange):
        self.popSize = popSize
        self.maxGen = maxGen
        self.T = T
        self.taskNumberRange = taskNumberRange
        self.population = []
        self.archive = []
        self.ref_point = []
        self.range_ = None
        self.objectNumber = 2
        self.evaluation_count = 0

        # Problem Notation - MEC Network
        self.edge_nodes = [
            {"id": 4, "type": "edge", "compute_capacity": 5, "radius": 100, "power_consumption": 0.8},
            {"id": 5, "type": "edge", "compute_capacity": 4, "radius": 90, "power_consumption": 0.7},
            {"id": 6, "type": "edge", "compute_capacity": 3, "radius": 80, "power_consumption": 0.6},
        ]
        self.cloud_node = {"id": 7, "type": "cloud", "compute_capacity": 50, "radius": 160, "power_consumption": 0}
        self.MEC_nodes = self.edge_nodes + [self.cloud_node]

        self.cloud_bandwidth = 1 * pow(10, 9)
        self.SeNBSet = []
        self.Bandwidth = 20
        self.N = 10
        self.w = (self.Bandwidth / self.N) * pow(10, 6)
        self.noisePower = pow(10, -176 / 10) * pow(10, -3)
        self.kk = pow(10, -11)
        self.totalSMDNumber = 0
        self.codeLength = 0
        self.H = 3

        self.readMECNetwork()
        self.M = self.SeNBSet.__len__()
        self.calculateInterference()
        self.calculateDataTransmissionRate()
        print("Total SMD number:", self.totalSMDNumber)
        print("Code length:", self.codeLength)

    def run(self):
        self.initializePopulation()
        self.evaluation_count = len(self.population)

        # Step 2: Generate uniform reference points
        W = self.uniform_point(self.popSize, self.objectNumber)

        # Step 3: Update reference points
        feasible_objs = [ind.fitness for ind in self.population if ind.isFeasible]
        self.archive, self.ref_point, self.range_ = self.update_ref_point(feasible_objs, W, None)

        # Step 4: Main optimization loop
        generation = 0
        while generation < self.maxGen:
            print(f'Generation-- {generation + 1}')

            # Step 5: Mating selection
            mating_pool = self.mating_selection(self.population, self.ref_point, self.range_)

            # Step 6: Generate offspring
            offspring = self.operator_ga(mating_pool)

            # Step 7: Update reference points
            offspring_feasible = [ind.fitness for ind in offspring if ind.isFeasible]
            if offspring_feasible:
                self.archive, self.ref_point, self.range_ = self.update_ref_point(
                    self.archive + offspring_feasible, W, self.range_)

            # Step 8-10: Environmental selection
            combined_population = self.population + offspring
            self.population, self.range_ = self.environmental_selection(
                combined_population, self.ref_point, self.range_, self.popSize)

            self.evaluation_count += len(offspring)
            generation += 1

        # Return Pareto front
        feasible_pop = [ind for ind in self.population if ind.isFeasible]
        EP_list = [ind.fitness for ind in feasible_pop]
        return EP_list

    def uniform_point(self, N, M):
        W = []
        if M == 2:
            beta = random.uniform(0.8, 1.5)  # 动态幂次控制，防止集中一侧
            for i in range(N):
                w1 = (i / (N - 1)) ** beta
                w2 = 1.0 - w1
                W.append([w1, w2])
        else:
            H = N - 1
            for i in range(H + 1):
                w = [(i / H), 1.0 - (i / H)]
                W.append(w)
        return np.array(W)

    def update_ref_point(self, archive, W, range_):
        """Update reference points based on archive"""
        if not archive:
            return [], W, range_

        archive_array = np.array(archive)

        # Remove duplicates and dominated solutions
        archive_array = self.remove_duplicates_dominated(archive_array)

        # 如果去重后没有解了，返回空
        if len(archive_array) == 0:
            return [], W, range_

        # Update range
        if range_ is not None:
            range_[0, :] = np.minimum(range_[0, :], np.min(archive_array, axis=0))
        else:
            range_ = np.array([np.min(archive_array, axis=0), np.max(archive_array, axis=0)])

        # Update reference points
        if len(archive_array) <= 1:
            ref_point = W
            choose = np.ones(len(archive_array), dtype=bool)  # 选择所有解
        else:
            t_archive = archive_array - range_[0, :]
            W_scaled = W * (range_[1, :] - range_[0, :])

            # Identify contributing solutions
            distance = self.cal_distance(t_archive, W_scaled)
            nearest_p = np.argmin(distance, axis=0)
            contributing_s = np.unique(nearest_p)

            nearest_w = np.argmin(distance, axis=1)
            valid_w = np.unique(nearest_w[contributing_s])

            # Select archive solutions
            choose = np.zeros(len(archive_array), dtype=bool)
            choose[contributing_s] = True

            # Add more solutions based on cosine similarity
            max_archive_size = min(3 * len(W), len(archive_array))
            if np.sum(choose) < max_archive_size:
                cosine_sim = self.calculate_cosine_similarity(t_archive)
                while np.sum(choose) < max_archive_size:
                    min_sim = float('inf')
                    best_idx = -1
                    for i in range(len(archive_array)):
                        if not choose[i]:
                            sim = np.max(cosine_sim[i, choose])
                            if sim < min_sim:
                                min_sim = sim
                                best_idx = i
                    if best_idx != -1:
                        choose[best_idx] = True
                    else:
                        break

            # Construct new reference point set
            ref_point_list = list(W_scaled[valid_w]) + list(t_archive[choose])
            ref_point = self.select_diverse_points(ref_point_list, len(W))

        return list(archive_array[choose]), ref_point, range_

    def environmental_selection(self, population, ref_point, range_, N):
        """Environmental selection"""
        # Calculate constraint violation
        CV = np.array([self.calculate_CV(ind) for ind in population])
        feasible_mask = CV == 0
        feasible_count = np.sum(feasible_mask)

        if feasible_count > N:
            # Case 1: Sufficient feasible solutions
            feasible_population = [population[i] for i in range(len(population)) if feasible_mask[i]]
            feasible_objs = np.array([ind.fitness for ind in feasible_population])

            front_no, max_f_no = self.nd_sort(feasible_objs, N)

            # 优先保留能耗较低的个体
            # energy_values = feasible_objs[:, 1]  # 第二个目标：能耗
            # sorted_idx = np.argsort(energy_values)
            # feasible_objs = feasible_objs[sorted_idx]

            next_pop = []
            for i in range(max_f_no - 1):
                next_pop.extend(np.where(front_no == i + 1)[0])

            # Handle last front
            last_front = np.where(front_no == max_f_no)[0]
            if len(next_pop) < N:
                last_objs = feasible_objs[last_front]
                choose = self.last_selection(last_objs, ref_point, range_, N - len(next_pop))
                next_pop.extend(last_front[choose])

            selected_population = [feasible_population[i] for i in next_pop[:N]]

            # Update range
            selected_objs = np.array([ind.fitness for ind in selected_population])
            range_ = np.array([np.min(selected_objs, axis=0), np.max(selected_objs, axis=0)])
        else:
            # Case 2: Insufficient feasible solutions
            sorted_indices = np.argsort(CV)
            selected_population = [population[i] for i in sorted_indices[:N]]

            selected_objs = np.array([ind.fitness for ind in selected_population])
            range_ = np.array([np.min(selected_objs, axis=0), np.max(selected_objs, axis=0)])

        return selected_population, range_

    def last_selection(self, pop_obj, ref_point, range_, K):
        """Select K solutions from the last front"""
        N = len(pop_obj)
        ref_point_array = np.array(ref_point) if isinstance(ref_point, list) else ref_point

        # Normalize
        normalized_pop = pop_obj - range_[0, :]

        # Calculate distance
        distance = self.cal_distance(normalized_pop, ref_point_array)
        convergence = np.min(distance, axis=1)

        sorted_indices = np.argsort(distance, axis=0)
        sorted_distance = distance[sorted_indices, np.arange(distance.shape[1])]

        remain = np.ones(N, dtype=bool)

        while np.sum(remain) > K:
            # Identify non-contributing solutions
            noncontributing = remain.copy()
            for j in range(sorted_indices.shape[1]):
                noncontributing[sorted_indices[0, j]] = False

            # Calculate metric values
            metric_base = np.sum(sorted_distance[0, :]) + np.sum(convergence[noncontributing])
            metric_values = np.full(N, np.inf)

            # Non-contributing solutions
            for p in np.where(noncontributing)[0]:
                metric_values[p] = metric_base - convergence[p]

            # Contributing solutions
            for p in np.where(~noncontributing & remain)[0]:
                temp_mask = sorted_indices[0, :] == p
                if np.any(temp_mask):
                    nc_idx = sorted_indices[1, temp_mask]
                    nc_mask = noncontributing[nc_idx]
                    metric_values[p] = (metric_base - np.sum(sorted_distance[0, temp_mask]) +
                                        np.sum(sorted_distance[1, temp_mask]) -
                                        np.sum(convergence[nc_idx[nc_mask]]))

            # Delete worst solution
            to_delete = np.argmin(metric_values)
            remain[to_delete] = False

            # Update sorted structures
            mask = sorted_indices != to_delete
            sorted_indices = sorted_indices[mask].reshape(sorted_indices.shape[0] - 1, -1)
            sorted_distance = sorted_distance[mask].reshape(sorted_distance.shape[0] - 1, -1)

        return np.where(remain)[0]

    def mating_selection(self, population, ref_point, range_):
        N = len(population)
        mating_pool = []
        CV = np.array([self.calculate_CV(ind) for ind in population])
        fitness = self.calculate_fitness_metric(population, ref_point, range_, CV)

        α = 0.5  # 平衡因子，越大越偏向时延，越小越偏向能耗（0.5 为平衡）

        for _ in range(N):
            i, j = random.sample(range(N), 2)
            if CV[i] < CV[j]:
                mating_pool.append(i)
            elif CV[i] > CV[j]:
                mating_pool.append(j)
            else:
                # 平衡选择
                if random.random() < α:  # α概率偏向时延（fitness值）
                    chosen = i if fitness[i] > fitness[j] else j
                else:  # (1-α)概率偏向能耗
                    chosen = i if population[i].fitness[1] < population[j].fitness[1] else j
                mating_pool.append(chosen)

        return mating_pool

    def operator_ga(self, mating_pool):
        """Genetic algorithm operators"""
        offspring = []

        for i in range(0, len(mating_pool) - 1, 2):
            parent1 = self.population[mating_pool[i]]
            parent2 = self.population[mating_pool[i + 1]]

            # Create offspring
            child1 = Individual()
            child2 = Individual()

            for gene1, gene2 in zip(parent1.chromosome, parent2.chromosome):
                smd1 = copy.deepcopy(gene1)
                smd2 = copy.deepcopy(gene2)
                self.reInitialize_WorkflowTaskSet_Schedule(smd1)
                self.reInitialize_WorkflowTaskSet_Schedule(smd2)
                child1.chromosome.append(smd1)
                child2.chromosome.append(smd2)

            # Crossover
            self.crossoverOperator(child1, child2)

            # Mutation
            self.mutantOperator(child1)
            self.mutantOperator(child2)

            # Evaluate
            self.calculateFitness(child1)
            self.calculateFitness(child2)

            offspring.extend([child1, child2])

        return offspring

    def calculate_fitness_metric(self, population, ref_point, range_, CV):
        """Calculate fitness metric for selection"""
        N = len(population)
        fitness = np.full(N, -np.inf)
        feasible_mask = CV == 0

        if np.any(feasible_mask) and ref_point is not None and len(ref_point) > 0:
            feasible_pop = [population[i] for i in range(N) if feasible_mask[i]]
            feasible_objs = np.array([ind.fitness for ind in feasible_pop])

            ref_point_array = np.array(ref_point) if isinstance(ref_point, list) else ref_point
            normalized_pop = feasible_objs - range_[0, :]

            distance = self.cal_distance(normalized_pop, ref_point_array)
            convergence = np.min(distance, axis=1)

            feasible_indices = np.where(feasible_mask)[0]
            fitness[feasible_indices] = -convergence

        return fitness

    def calculate_CV(self, ind):
        """Calculate constraint violation (0 for feasible)"""
        return 0.0  # Assume all solutions are feasible

    def nd_sort(self, objs, N):
        """Non-dominated sorting"""
        n = len(objs)
        domination_count = np.zeros(n)
        dominated_solutions = [[] for _ in range(n)]
        front_no = np.zeros(n, dtype=int)

        for i in range(n):
            for j in range(i + 1, n):
                if self.dominates(objs[i], objs[j]):
                    dominated_solutions[i].append(j)
                    domination_count[j] += 1
                elif self.dominates(objs[j], objs[i]):
                    dominated_solutions[j].append(i)
                    domination_count[i] += 1

        current_front = np.where(domination_count == 0)[0]
        front_no[current_front] = 1
        max_front = 1

        while len(current_front) > 0:
            next_front = []
            for i in current_front:
                for j in dominated_solutions[i]:
                    domination_count[j] -= 1
                    if domination_count[j] == 0:
                        next_front.append(j)

            if next_front:
                max_front += 1
                front_no[next_front] = max_front
                current_front = next_front
            else:
                break

        return front_no, max_front

    def dominates(self, obj1, obj2):
        """Check if obj1 dominates obj2"""
        return np.all(obj1 <= obj2) and np.any(obj1 < obj2)

    def cal_distance(self, pop, ref):
        """Calculate Euclidean distance matrix"""
        pop = np.atleast_2d(pop)
        ref = np.atleast_2d(ref)
        dist = np.sqrt(((pop[:, np.newaxis, :] - ref[np.newaxis, :, :]) ** 2).sum(axis=2))
        return dist

    def calculate_cosine_similarity(self, vectors):
        """Calculate cosine similarity matrix"""
        norm = np.linalg.norm(vectors, axis=1, keepdims=True)
        norm[norm == 0] = 1
        normalized = vectors / norm
        return np.dot(normalized, normalized.T)

    def select_diverse_points(self, points, K):
        """Select K diverse points based on cosine similarity"""
        if len(points) <= K:
            return np.array(points)

        points_array = np.array(points)
        cosine_sim = self.calculate_cosine_similarity(points_array)

        selected = [0]
        for _ in range(K - 1):
            min_sim = float('inf')
            best_idx = -1
            for i in range(len(points)):
                if i not in selected:
                    sim = np.max([cosine_sim[i, j] for j in selected])
                    if sim < min_sim:
                        min_sim = sim
                        best_idx = i
            if best_idx != -1:
                selected.append(best_idx)
            else:
                break

        return points_array[selected]

    def remove_duplicates_dominated(self, objs):
        """Remove duplicate and dominated solutions"""
        unique_objs = []
        for obj in objs:
            is_dominated = False
            to_remove = []
            for i, existing in enumerate(unique_objs):
                if np.allclose(obj, existing):
                    is_dominated = True
                    break
                if self.dominates(existing, obj):
                    is_dominated = True
                    break
                if self.dominates(obj, existing):
                    to_remove.append(i)

            for i in reversed(to_remove):
                unique_objs.pop(i)

            if not is_dominated:
                unique_objs.append(obj)

        return np.array(unique_objs) if unique_objs else np.array([]).reshape(0, self.objectNumber)

    # ========== Problem-specific methods (from original MOEAD) ==========

    def initializePopulation(self):
        for i in range(self.popSize):
            ind = Individual()
            for senb in self.SeNBSet:
                for smd in senb.SMDSet:
                    temp_smd = copy.deepcopy(smd)
                    connected_node = self.getConnectedEdgeNode(temp_smd.coordinate, senb.coordinate)

                    for j in range(temp_smd.workflow.taskNumber):
                        pos_range = [1, 2, 3, connected_node["id"], 7]
                        temp_smd.workflow.position.append(random.choice(pos_range))

                    temp_smd.workflow.sequence = self.initializeWorkflowSequence(temp_smd.workflow)
                    ind.chromosome.append(temp_smd)
            self.calculateFitness(ind)
            self.population.append(ind)

    def getConnectedEdgeNode(self, smd_coordinate, senb_coordinate):
        distance_to_senb = self.getDistance(smd_coordinate, senb_coordinate)
        available_nodes = []
        for node in self.MEC_nodes:
            if distance_to_senb <= node["radius"]:
                available_nodes.append(node)
        return random.choice(available_nodes)

    def crossoverOperator(self, ind_k, ind_l):
        for i in range(self.totalSMDNumber):
            gene_1 = ind_k.chromosome[i]
            gene_2 = ind_l.chromosome[i]
            cpt = random.randint(0, len(gene_1.workflow.position) - 1)
            cPart_1 = []
            cPart_2 = []

            for j in range(0, cpt):
                gene_1.workflow.position[j], gene_2.workflow.position[j] = gene_2.workflow.position[j], \
                gene_1.workflow.position[j]
                cPart_1.append(gene_1.workflow.sequence[j])
                cPart_2.append(gene_2.workflow.sequence[j])

            for j in range(len(cPart_1)):
                gene_2.workflow.sequence.remove(cPart_1[j])
                gene_1.workflow.sequence.remove(cPart_2[j])

            gene_1.workflow.sequence = cPart_2 + gene_1.workflow.sequence
            gene_2.workflow.sequence = cPart_1 + gene_2.workflow.sequence

    def mutantOperator(self, ind):
        for gene in ind.chromosome:
            rnd_SMD = myRandom.get_0to1_RandomNumber()
            # 保持个体整体变异概率不变
            if rnd_SMD < 1.0 / self.totalSMDNumber:
                adaptive_rate = max(0.05, 1.0 / math.sqrt(gene.workflow.taskNumber))  # 自适应
                for i in range(len(gene.workflow.position)):
                    rnd_bit = myRandom.get_0to1_RandomNumber()
                    if rnd_bit < adaptive_rate:  # 自适应变异概率
                        pos = gene.workflow.position[i]
                        rand = [1, 2, 3, 4, 5, 6, 7]
                        rand.remove(pos)
                        gene.workflow.position[i] = random.choice(rand)

                r = random.randint(1, gene.workflow.sequence.__len__() - 2)
                formerSetPoint = []
                rearSetPoint = []

                for i in range(0, gene.workflow.sequence.__len__() - 1):
                    formerSetPoint.append(gene.workflow.sequence[i])
                    if set(gene.workflow.taskSet[r].preTaskSet).issubset(set(formerSetPoint)):
                        break

                for j in range(gene.workflow.sequence.__len__() - 1, -1, -1):
                    rearSetPoint.append(gene.workflow.sequence[j])
                    if set(gene.workflow.taskSet[r].sucTaskSet).issubset(set(rearSetPoint)):
                        break

                rnd_insert_pt = random.randint(i + 1, j - 1)
                gene.workflow.sequence.remove(r)
                gene.workflow.sequence.insert(rnd_insert_pt, r)

    def calculateFitness(self, ind):
        ind.fitness = []
        time = []
        energy = []
        for gene in ind.chromosome:
            smd = gene
            self.calculateWorkflowTimeEnergy(smd, smd.workflow)
            time.append(smd.workflow.schedule.T_total)
            energy.append(smd.workflow.schedule.E_total)
        ind.fitness.append(np.average(time))
        ind.fitness.append(np.average(energy))

    # def calculateFitness(self, ind):
    #     ind.fitness = []
    #     time = []
    #     energy = []
    #
    #     for gene in ind.chromosome:
    #         smd = gene
    #         self.calculateWorkflowTimeEnergy(smd, smd.workflow)
    #         time.append(smd.workflow.schedule.T_total)
    #         energy.append(smd.workflow.schedule.E_total)
    #
    #     # 这里通过调整权重来加重能耗的影响
    #     time_weight = 0.3  # 减小时间的权重
    #     energy_weight = 0.7  # 增大能耗的权重
    #
    #     # 计算适应度时，能耗的影响比时间更大
    #     ind.fitness.append(np.average(time) * time_weight)
    #     ind.fitness.append(np.average(energy) * energy_weight)

    def calculateWorkflowTimeEnergy(self, smd, workflow):
        workflow.schedule.TimeEnergy = []
        workflow.schedule.T_total = None
        workflow.schedule.E_total = 0

        # 初始化所有执行单元的时间点
        for node in self.MEC_nodes:
            workflow.schedule.MECTP[node["id"]] = [0]
        workflow.schedule.cloudTP = [0]  # 云中心时间点

        for i in range(len(workflow.sequence)):
            taskId = workflow.sequence[i]
            pos = workflow.position[i]
            task = workflow.taskSet[taskId]
            task.exePosition = pos

            if pos >= 4:  # MEC节点或云中心执行
                task.islocal = False
                mec_node = None
                for node in self.MEC_nodes:
                    if node["id"] == pos:
                        mec_node = node
                        break

                if mec_node is None:
                    continue

                # 入口任务
                if task.id == workflow.entryTask:
                    task.RT_i_l = task.ST_i_l = task.FT_i_l = 0

                    # 无线发送到边缘节点
                    task.RT_i_ws = task.ST_i_ws = 0.0
                    task.FT_i_ws = task.ST_i_ws + task.d_i_j_k / smd.R_i_j

                    if mec_node["type"] == "cloud":
                        # 云中心执行：额外增加有线传输时间
                        task.RT_i_c = task.ST_i_c = task.FT_i_ws
                        # 有线传输到云中心
                        task.FT_i_c = task.ST_i_c + task.d_i_j_k / self.cloud_bandwidth
                        # 云中心计算
                        task.ST_i_cloud = task.FT_i_c
                        task.FT_i_cloud = task.ST_i_cloud + task.c_i_j_k / mec_node["compute_capacity"]
                        # 接收就绪时间（云中心执行无返回传输时间）
                        task.RT_i_wr = task.ST_i_wr = task.FT_i_cloud
                        task.FT_i_wr = task.ST_i_wr  # 无返回数据传输

                        workflow.schedule.wsTP.append(task.FT_i_ws)
                        workflow.schedule.MECTP[pos].append(task.FT_i_c)
                        workflow.schedule.cloudTP.append(task.FT_i_cloud)
                    else:
                        # 边缘节点执行
                        task.RT_i_c = task.ST_i_c = task.FT_i_ws = 0
                        task.RT_i_ws = task.ST_i_ws = 0.0
                        task.FT_i_ws = task.ST_i_ws + task.d_i_j_k / smd.R_i_j
                        task.RT_i_c = task.ST_i_c = task.FT_i_ws
                        task.FT_i_c = task.ST_i_c + task.c_i_j_k / mec_node["compute_capacity"]
                        task.RT_i_wr = task.ST_i_wr = task.FT_i_c
                        task.FT_i_wr = task.ST_i_wr + task.o_i_j_k / smd.R_i_j

                        workflow.schedule.wsTP.append(task.FT_i_ws)
                        workflow.schedule.MECTP[pos].append(task.FT_i_c)
                        workflow.schedule.wrTP.append(task.FT_i_wr)
                else:  # 非入口任务
                    task.ST_i_l = float("inf")
                    task.FT_i_l = float("inf")

                    # 无线发送就绪时间
                    task.RT_i_ws = self.get_RT_i_ws(task, workflow)
                    if workflow.schedule.wsTP[-1] < task.RT_i_ws:
                        task.ST_i_ws = task.RT_i_ws
                    else:
                        task.ST_i_ws = workflow.schedule.wsTP[-1]
                    task.FT_i_ws = task.ST_i_ws + task.d_i_j_k / smd.R_i_j
                    workflow.schedule.wsTP.append(task.FT_i_ws)

                    if mec_node["type"] == "cloud":
                        # 云中心执行
                        task.RT_i_c = self.get_RT_i_c(task, workflow)
                        if workflow.schedule.MECTP[pos][-1] < task.RT_i_c:
                            task.ST_i_c = task.RT_i_c
                        else:
                            task.ST_i_c = workflow.schedule.MECTP[pos][-1]
                        # 有线传输到云中心
                        task.FT_i_c = task.ST_i_c + task.d_i_j_k / self.cloud_bandwidth
                        workflow.schedule.MECTP[pos].append(task.FT_i_c)

                        # 云中心计算
                        task.RT_i_cloud = max(task.FT_i_c, self.get_RT_i_cloud(task, workflow))
                        if workflow.schedule.cloudTP[-1] < task.RT_i_cloud:
                            task.ST_i_cloud = task.RT_i_cloud
                        else:
                            task.ST_i_cloud = workflow.schedule.cloudTP[-1]
                        task.FT_i_cloud = task.ST_i_cloud + task.c_i_j_k / mec_node["compute_capacity"]
                        workflow.schedule.cloudTP.append(task.FT_i_cloud)

                        # 接收就绪时间
                        task.RT_i_wr = task.ST_i_wr = task.FT_i_cloud
                        task.FT_i_wr = task.ST_i_wr  # 无返回传输
                    else:
                        # 边缘节点执行
                        task.RT_i_c = self.get_RT_i_c(task, workflow)
                        if workflow.schedule.MECTP[pos][-1] < task.RT_i_c:
                            task.ST_i_c = task.RT_i_c
                        else:
                            task.ST_i_c = workflow.schedule.MECTP[pos][-1]
                        task.FT_i_c = task.ST_i_c + task.c_i_j_k / mec_node["compute_capacity"]
                        workflow.schedule.MECTP[pos].append(task.FT_i_c)

                        task.RT_i_wr = task.FT_i_c
                        if workflow.schedule.wrTP[-1] < task.RT_i_wr:
                            task.ST_i_wr = task.RT_i_wr
                        else:
                            task.ST_i_wr = workflow.schedule.wrTP[-1]
                        task.FT_i_wr = task.ST_i_wr + task.o_i_j_k / smd.R_i_j
                        workflow.schedule.wrTP.append(task.FT_i_wr)

                # 计算能耗（云中心执行能耗为0）
                if mec_node["type"] != "cloud":
                    task.energy += smd.pws_i_j * (task.FT_i_ws - task.ST_i_ws)
                    task.energy += smd.pwr_i_j * (task.FT_i_wr - task.ST_i_wr)
                    # 边缘节点执行能耗
                    execution_time = task.FT_i_c - task.ST_i_c
                    task.energy += mec_node["power_consumption"] * execution_time

                workflow.schedule.E_total += task.energy
            else:  # 本地执行
                task.islocal = True
                task.RT_i_ws = task.RT_i_c = task.RT_i_wr = 0.0
                task.ST_i_ws = task.ST_i_c = task.ST_i_wr = 0.0
                task.FT_i_ws = task.FT_i_c = task.FT_i_wr = 0.0

                if task.id == workflow.entryTask:
                    task.RT_i_l = task.ST_i_l = 0
                    task.FT_i_l = task.ST_i_l + task.c_i_j_k / smd.coreCC[pos]
                else:
                    task.RT_i_l = self.get_RT_i_l(task, workflow)
                    if task.RT_i_l > workflow.schedule.coreTP[pos][-1]:
                        task.ST_i_l = task.RT_i_l
                    else:
                        task.ST_i_l = workflow.schedule.coreTP[pos][-1]
                    task.FT_i_l = task.ST_i_l + task.c_i_j_k / smd.coreCC[pos]

                workflow.schedule.coreTP[pos].append(task.FT_i_l)
                task.energy = smd.pcc_i_j[pos] * (task.FT_i_l - task.ST_i_l)
                workflow.schedule.E_total += task.energy

            workflow.schedule.S[pos].append(task.id)

        # 计算工作流总时间
        exit_task = workflow.taskSet[workflow.exitTask]
        if exit_task.islocal:
            workflow.schedule.T_total = exit_task.FT_i_l
        elif exit_task.exePosition == 7:  # 云中心执行
            workflow.schedule.T_total = exit_task.FT_i_cloud
        else:  # 边缘节点执行
            workflow.schedule.T_total = exit_task.FT_i_wr

        workflow.schedule.TimeEnergy.append(workflow.schedule.T_total)
        workflow.schedule.TimeEnergy.append(workflow.schedule.E_total)

    def get_RT_i_ws(self, task, workflow):
        if task.id == workflow.entryTask:
            return 0.0
        else:
            pre_max = []
            for pre_taskId in task.preTaskSet:
                if workflow.taskSet[pre_taskId].islocal == True:
                    pre_max.append(workflow.taskSet[pre_taskId].FT_i_l)
                else:
                    pre_max.append(workflow.taskSet[pre_taskId].FT_i_ws)
            return max(pre_max)

    def get_RT_i_c(self, task, workflow):
        pre_max = []
        for pre_taskId in task.preTaskSet:
            pre_max.append(workflow.taskSet[pre_taskId].FT_i_c)
        return max(task.FT_i_ws, max(pre_max))

    def get_RT_i_l(self, task, workflow):
        if task.id == workflow.entryTask:
            return 0.0
        else:
            pre_max = []
            for pre_taskId in task.preTaskSet:
                if workflow.taskSet[pre_taskId].islocal == True:
                    pre_max.append(workflow.taskSet[pre_taskId].FT_i_l)
                else:
                    pre_max.append(workflow.taskSet[pre_taskId].FT_i_wr)
            return max(pre_max)

    def get_RT_i_cloud(self, task, workflow):
        pre_max = []
        for pre_taskId in task.preTaskSet:
            pre_task = workflow.taskSet[pre_taskId]
            if pre_task.exePosition == 7:
                pre_max.append(pre_task.FT_i_cloud)
            else:
                pre_max.append(pre_task.FT_i_c)
        return max(pre_max) if pre_max else 0

    def reInitialize_WorkflowTaskSet_Schedule(self, smd):
        for task in smd.workflow.taskSet:
            self.reInitializeTaskSet(task)
        self.reInitializeSchedule(smd.workflow.schedule)

    def reInitializeTaskSet(self, task):
        task.islocal = None
        task.exePosition = None
        task.RT_i_l = task.ST_i_l = task.FT_i_l = None
        task.RT_i_ws = task.RT_i_c = task.RT_i_wr = None
        task.ST_i_ws = task.ST_i_c = task.ST_i_wr = None
        task.FT_i_ws = task.FT_i_c = task.FT_i_wr = None
        task.energy = 0

    def reInitializeSchedule(self, schedule):
        schedule.S = {1: [], 2: [], 3: [], 4: [], 5: [], 6: [], 7: []}
        schedule.coreTP = {1: [0], 2: [0], 3: [0]}
        schedule.wsTP = [0]
        schedule.MECTP = {}
        for node in self.MEC_nodes:
            schedule.MECTP[node["id"]] = [0]
        schedule.cloudTP = [0]
        schedule.wrTP = [0]
        schedule.T_total = None
        schedule.E_total = 0
        schedule.TimeEnergy = []

    def initializeWorkflowSequence(self, workflow):
        S = []
        R = []
        T = []
        R.append(workflow.entryTask)
        for task in workflow.taskSet:
            T.append(task.id)
        T.remove(workflow.entryTask)

        while T != []:
            for t in T:
                if set(workflow.taskSet[t].preTaskSet).issubset(set(R)):
                    if t not in S:
                        S.append(t)
            ti = random.choice(S)
            S.remove(ti)
            T.remove(ti)
            R.append(ti)
        return R

    def calculateInterference(self):
        for i in range(self.M):
            for j in range(self.SeNBSet[i].SMDNumber):
                I_i_j = 0
                for m in range(self.M):
                    if self.SeNBSet[m] != self.SeNBSet[i]:
                        for k in range(self.SeNBSet[m].SMDNumber):
                            if self.SeNBSet[m].SMDSet[k].channel == self.SeNBSet[i].SMDSet[j].channel:
                                g_i_m_k = self.getChannelGain(self.SeNBSet[m].SMDSet[k].coordinate,
                                                              self.SeNBSet[i].coordinate)
                                I_i_j += self.SeNBSet[m].SMDSet[k].pws_i_j * g_i_m_k
                self.SeNBSet[i].SMDSet[j].I_i_j = I_i_j

    def calculateDataTransmissionRate(self):
        self.calculateChannelGain()
        for i in range(self.M):
            for j in range(self.SeNBSet[i].SMDNumber):
                log_v = 1 + (self.SeNBSet[i].SMDSet[j].pws_i_j * self.SeNBSet[i].SMDSet[j].g_i_j) / \
                        (self.noisePower + self.SeNBSet[i].SMDSet[j].I_i_j)
                self.SeNBSet[i].SMDSet[j].R_i_j = self.w * math.log(log_v, 2)

    def calculateChannelGain(self):
        for i in range(self.M):
            for j in range(self.SeNBSet[i].SMDNumber):
                self.SeNBSet[i].SMDSet[j].g_i_j = self.getChannelGain(
                    self.SeNBSet[i].SMDSet[j].coordinate, self.SeNBSet[i].coordinate)

    def getChannelGain(self, U_i_j, S_i):
        distance = self.getDistance(U_i_j, S_i)
        channelGain = pow(distance, -4)
        return channelGain

    def getDistance(self, point1, point2):
        return np.sqrt(np.sum(np.square([point1[i] - point2[i] for i in range(2)])))

    def getWorkflow(self, filename):
        wf = Workflow()
        with open(filename, 'r') as readFile:
            for line in readFile:
                task = Task()
                s = line.splitlines()
                s = s[0].split(':')
                predecessor = s[0]
                id = s[1]
                successor = s[2]
                if predecessor != '':
                    predecessor = predecessor.split(',')
                    for pt in predecessor:
                        task.preTaskSet.append(int(pt))
                else:
                    wf.entryTask = int(id)
                task.id = int(id)
                if successor != '':
                    successor = successor.split(',')
                    for st in successor:
                        task.sucTaskSet.append(int(st))
                else:
                    wf.exitTask = int(id)
                wf.taskSet.append(task)
        return wf

    def readMECNetwork(self):
        file_SMD_task_cpu = open(self.getCurrentPath() + '\\' + self.taskNumberRange +
                                 '\SMD_Task_CPU_Cycles_Number.txt', 'r')
        file_SMD_task_data = open(self.getCurrentPath() + '\\' + self.taskNumberRange +
                                  '\SMD_Task_Data_Size.txt', 'r')
        file_SMD_output_task_data = open(self.getCurrentPath() + '\\' + self.taskNumberRange +
                                         '\SMD_Task_Output_Data_Size.txt', 'r')

        SeNB_count = -1
        with open(self.getCurrentPath() + '\\' + self.taskNumberRange + '\MEC_Network.txt', 'r') as readFile:
            for line in readFile:
                if line == '---file end---\n':
                    break
                elif line == 'SeNB:\n':
                    SeNB_count += 1
                    senb = SeNB()
                    if readFile.readline() == 'Coordinate:\n':
                        SeNB_crd = readFile.readline()
                        SeNB_crd = SeNB_crd.splitlines()
                        SeNB_crd = SeNB_crd[0].split('  ')
                        senb.coordinate.append(float(SeNB_crd[0]))
                        senb.coordinate.append(float(SeNB_crd[1]))

                        if readFile.readline() == 'SMD number:\n':
                            senb.SMDNumber = int(readFile.readline())

                        for line1 in readFile:
                            if line1 == '---SeNB end---\n':
                                break
                            elif line1 == 'SMD:\n':
                                self.totalSMDNumber += 1
                                smd = SMD()

                                if readFile.readline() == 'Coordinate:\n':
                                    SMD_crd = readFile.readline()
                                    SMD_crd = SMD_crd.splitlines()
                                    SMD_crd = SMD_crd[0].split('  ')
                                    smd.coordinate.append(float(SMD_crd[0]))
                                    smd.coordinate.append(float(SMD_crd[1]))

                                if readFile.readline() == 'Computation capacity:\n':
                                    SMD_cc = readFile.readline()
                                    SMD_cc = SMD_cc.splitlines()
                                    SMD_cc = SMD_cc[0].split('  ')
                                    smd.coreCC[1] = float(SMD_cc[0])
                                    smd.coreCC[2] = float(SMD_cc[1])
                                    smd.coreCC[3] = float(SMD_cc[2])

                                if readFile.readline() == 'The number of task:\n':
                                    taskNumber = int(readFile.readline())
                                    SeNB_directory = "SeNB-" + str(SeNB_count) + "\\t" + str(taskNumber) + ".txt"
                                    wf_directory = self.getCurrentPath() + "\workflowSet\\" + SeNB_directory
                                    smd.workflow = self.getWorkflow(wf_directory)
                                    smd.workflow.taskNumber = taskNumber
                                    self.codeLength += taskNumber

                                    for task in smd.workflow.taskSet:
                                        task.c_i_j_k = float(file_SMD_task_cpu.readline())
                                        task.d_i_j_k = float(file_SMD_task_data.readline()) * 1024
                                        task.o_i_j_k = float(file_SMD_output_task_data.readline()) * 1024

                                if readFile.readline() == 'Channel:\n':
                                    channel = readFile.readline()
                                    smd.channel = int(channel)

                                senb.SMDSet.append(smd)
                    self.SeNBSet.append(senb)

        file_SMD_task_data.close()
        file_SMD_task_cpu.close()
        file_SMD_output_task_data.close()

    def getCurrentPath(self):
        return os.path.dirname(os.path.realpath(__file__))


# ========== Data Structures ==========

class Individual:
    def __init__(self):
        self.chromosome = []
        self.fitness = []
        self.isFeasible = True
        self.temp_fitness = None
        self.distance = 0.0
        self.rank = None
        self.S_p = []
        self.n = 0


class SeNB:
    def __init__(self):
        self.coordinate = []
        self.SMDNumber = 0
        self.SMDSet = []


class SMD:
    def __init__(self):
        self.coordinate = []
        self.workflow = Workflow()
        self.channel = None
        self.g_i_j = None
        self.R_i_j = None
        self.I_i_j = None
        self.coreCC = {1: None, 2: None, 3: None}
        self.pcc_i_j = {1: 4, 2: 2, 3: 1}
        self.pws_i_j = 0.5
        self.pwr_i_j = 0.1


class Workflow:
    def __init__(self):
        self.entryTask = None
        self.exitTask = None
        self.position = []
        self.sequence = []
        self.taskNumber = None
        self.taskSet = []
        self.schedule = Schedule()


class Schedule:
    def __init__(self):
        self.taskSet = {}
        self.S = {1: [], 2: [], 3: [], 4: [], 5: [], 6: [], 7: []}
        self.coreTP = {1: [0], 2: [0], 3: [0]}
        self.wsTP = [0]
        self.MECTP = {}
        self.cloudTP = [0]
        self.wrTP = [0]
        self.T_total = None
        self.E_total = 0
        self.TimeEnergy = []


class Task:
    def __init__(self):
        self.id = None
        self.islocal = None
        self.preTaskSet = []
        self.sucTaskSet = []
        self.exePosition = None
        self.actualFre = 1
        self.c_i_j_k = None
        self.d_i_j_k = None
        self.o_i_j_k = None

        self.RT_i_l = None
        self.RT_i_ws = None
        self.RT_i_c = None
        self.RT_i_wr = None
        self.RT_i_cloud = None

        self.ST_i_l = None
        self.ST_i_ws = None
        self.ST_i_c = None
        self.ST_i_wr = None
        self.ST_i_cloud = None

        self.FT_i_l = None
        self.FT_i_ws = None
        self.FT_i_c = None
        self.FT_i_wr = None
        self.FT_i_cloud = None
        self.energy = 0


class Pareto:
    def __init__(self):
        self.chromosome = None
        self.fitness = []
        self.temp_fitness = None