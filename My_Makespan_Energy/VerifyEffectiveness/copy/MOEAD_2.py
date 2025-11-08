import random, copy, math, turtle, os,sys
sys.path.append("./")
from Tool import myRandom, myFileOperator
import matplotlib.pyplot as plt
import numpy as np

class MOEAD:
    def __init__(self, popSize, maxGen, T, taskNumberRange):
        self.popSize = popSize
        self.maxGen = maxGen
        self.taskNumberRange = taskNumberRange
        self.VT = {}  # 权重向量集合
        self.B = {}  # 权向量的邻居
        self.T = T  # 邻居数量
        self.population = []
        self.Z = []  # 参考点
        self.objectNumber = 2
        self.F_rank = []  # 将种群非支配排序分层, 用种群中的个体的下标来表示，一个元素表示第一层,下标从1开始

        self.PF_history = []            #每一代的历史最优Pareto Front
        self.EP = []                    #保存当前代的历史非支配解
        #---------------------------------------Problem Notation----------------------------------------
        self.MEC_nodes = [
            {"id": 4, "type": "edge", "compute_capacity": 5, "radius": 90, "power_consumption": 0.8},
            {"id": 5, "type": "edge", "compute_capacity": 4, "radius": 85, "power_consumption": 0.7},
            {"id": 6, "type": "edge", "compute_capacity": 3, "radius": 80, "power_consumption": 0.6},
            {"id": 7, "type": "cloud", "compute_capacity": 50, "radius": 160, "power_consumption": 0}  # 添加云中心功耗参数
        ]
        self.cloud_bandwidth = 1 * pow(10, 9)
        self.SeNBSet = []

        self.Bandwidth = 20             #Bandwidth
        self.N = 10             #Number of channel
        self.w = (self.Bandwidth / self.N) * pow(10, 6)  #The bandwidth of channel
        self.noisePower = pow(10, -176/10)*pow(10, -3)  #The background noise power (50dBm = 100w)
        self.kk = pow(10, -11)  #It is a coefficient depending on the chip architecture
        self.totalSMDNumber = 0  # The total number of SMD in the network system
        self.codeLength = 0

        self.H = 3  # The number of the core in a SMD.
        self.readMECNetwork()
        self.M = self.SeNBSet.__len__()
        self.calculateInterference()
        self.calculateDataTransmissionRate()
        print("The total SMD number: ", self.totalSMDNumber)
        print("The code length: ", self.codeLength)

        self.a = [0.2, 0.5, 0.8, 1]  # The frequency scaling factors
        self.M = len(self.a)  # M different frequency levels.


    def run(self):
        self.initializeWeightVectorAndNeighbor()
        self.initializePopulation()
        self.initializeReferencePoint()
        self.fast_non_dominated_sort(self.population)
        self.initializeEP(self.F_rank[1])

        t = 1
        while (t <= self.maxGen):
            print('Generation--', t)

            for i in range(self.popSize):
                y_ = self.reproduction(i)
                self.DVFS_Schedule_Algorithm(y_)
                self.updateReferencePoint(y_)
                self.updateNeighborSolutions(i, y_)
                self.update_EP_FromElement(self.EP, y_)
            t += 1



        for ep in self.EP:
            ep.temp_fitness = ep.fitness[0]
        test_fast = sorted(self.EP, key=lambda Individual: Individual.temp_fitness)
        EP_list = [copy.deepcopy(ind.fitness) for ind in test_fast]
        return EP_list  # 返回最终的非支配解集

    """
        **********************************************run**********************************************
    """

    def DVFS_Schedule_Algorithm(self, ind):
        for gene in ind.chromosome:
            workflow = gene.workflow
            schedule = gene.workflow.schedule
            for coreId in schedule.S:
                if coreId < 4 and schedule.S[coreId] != []:
                    for i in range(len(schedule.S[coreId])):
                        taskId = schedule.S[coreId][i]
                        vi = workflow.taskSet[taskId]
                        flag = 0
                        m = 1
                        while flag == 0 and m < self.M:
                            FT_i_new = self.DVFS_calculateNewFinishTime(m, vi)
                            if i != (len(schedule.S[coreId]) - 1):  # There is next task v_j on the same core
                                vj = workflow.taskSet[schedule.S[coreId][i + 1]]
                                lim1 = vj.ST_i_l
                            else:
                                lim1 = workflow.schedule.T_total
                            if vi.id != workflow.exitTask:
                                lim2 = self.get_min_Succ_ST(workflow, vi)
                            else:
                                lim2 = workflow.schedule.T_total
                            if FT_i_new <= lim1 and FT_i_new <= lim2:
                                flag = 1
                                vi.actualFre = m
                                vi.FT_i_l = FT_i_new
                                schedule.E_total -= vi.energy
                                vi.energy = self.a[vi.actualFre - 1] * vi.energy
                                schedule.E_total +=  vi.energy
                                schedule.TimeEnergy[1] = schedule.E_total
                                break
                            m += 1

    def DVFS_calculateNewFinishTime(self, m, task):
        return task.ST_i_l + (task.FT_i_l - task.ST_i_l) / self.a[m-1]


    def get_min_Succ_ST(self, workflow, task):
        minST = []
        for succ_taskId in task.sucTaskSet:
            succ = workflow.taskSet[succ_taskId]
            if succ.islocal == True:
                minST.append(succ.ST_i_l)
            else:
                minST.append(succ.ST_i_ws)
        return min(minST)


    def initializeEP(self, F_rank):
        for ind in F_rank:
            self.EP.append(copy.deepcopy(ind))


    def initializeWeightVectorAndNeighbor(self):
        H = self.popSize - 1
        for i in range(0, H+1):
            w = []
            w1 = i/H - 0.0
            w2 = 1.0 - i/H
            w.append(w1)
            w.append(w2)
            self.VT[i] = w

        for i in self.VT.keys():
            distance = []
            for j in self.VT.keys():
                if(i != j):
                    tup = (j, self.getDistance(self.VT[i], self.VT[j]))
                    distance.append(tup)
            distance= sorted(distance, key=lambda x:x[1])
            neighbor = []
            for j in range(self.T):
                neighbor.append(distance[j][0])
            self.B[i] = neighbor



    def initializePopulation(self):
        for i in range(int(self.popSize)):
            ind = Individual()
            for senb in self.SeNBSet:
                for smd in senb.SMDSet:
                    temp_smd = copy.deepcopy(smd)
                    connected_node = self.getConnectedMECNode(temp_smd.coordinate, senb.coordinate)

                    for j in range(temp_smd.workflow.taskNumber):
                        task = temp_smd.workflow.taskSet[j]

                        # 计算本地执行时间（三个核心的平均值）
                        T_local = (task.c_i_j_k / temp_smd.coreCC[1] +
                                   task.c_i_j_k / temp_smd.coreCC[2] +
                                   task.c_i_j_k / temp_smd.coreCC[3]) / 3

                        # 计算不同MEC节点的执行时间
                        T_mec_options = []
                        for node in self.MEC_nodes:
                            if node["type"] == "edge" and node["id"] in [4, 5, 6]:
                                # 边缘节点执行：无线传输 + 边缘计算 + 无线返回
                                T_edge = (task.d_i_j_k + task.o_i_j_k) / temp_smd.R_i_j + task.c_i_j_k / node[
                                    "compute_capacity"]
                                T_mec_options.append((node["id"], T_edge))
                            elif node["type"] == "cloud" and node["id"] == 7:
                                # 云中心执行：无线传输 + 有线传输 + 云计算
                                T_cloud = (
                                                  task.d_i_j_k + task.o_i_j_k) / temp_smd.R_i_j + task.d_i_j_k / self.cloud_bandwidth + task.c_i_j_k / \
                                          node["compute_capacity"]
                                T_mec_options.append((node["id"], T_cloud))

                        # 选择最优的MEC节点
                        best_mec_id = min(T_mec_options, key=lambda x: x[1])[0] if T_mec_options else None

                        if best_mec_id and T_mec_options[best_mec_id - 4][1] <= T_local:
                            temp_smd.workflow.position.append(best_mec_id)
                        else:
                            temp_smd.workflow.position.append(random.randint(1, 3))

                    temp_smd.workflow.sequence = self.initializeWorkflowSequence(temp_smd.workflow)
                    ind.chromosome.append(temp_smd)
            self.calculateFitness(ind)
            self.population.append(ind)

    def getConnectedMECNode(self, smd_coordinate, senb_coordinate):
        distance_to_senb = self.getDistance(smd_coordinate, senb_coordinate)
        # 收集所有在覆盖范围内的 MEC 节点
        available_nodes = []
        for node in self.MEC_nodes:
            # 简化处理：MEC 节点部署在 SeNB 位置，所以距离与到 SeNB 的距离相同
            if distance_to_senb <= node["radius"]:
                available_nodes.append(node)
        # 随机选择一个可用的节点
        if available_nodes:
            return random.choice(available_nodes)
        else:
            return None  # 如果没有可用的节点，返回 None


    def initializeReferencePoint(self):
        fitness_1 = [] #存储所有个体的第一个适应度值
        fitness_2 = [] #存储所有个体的第二个适应度值
        for ind in self.population:
            fitness_1.append(ind.fitness[0])
            fitness_2.append(ind.fitness[1])
        self.Z.append(min(fitness_1))
        self.Z.append(min(fitness_2))


    def reproduction(self, i):
        k = random.choice(self.B[i])
        l = random.choice(self.B[i])
        ind_k = Individual()
        ind_l = Individual()
        for gene in self.population[k].chromosome:
            smd_k = copy.deepcopy(gene)
            self.reInitialize_WorkflowTaskSet_Schedule(smd_k)
            ind_k.chromosome.append(smd_k)

        for gene in self.population[l].chromosome:
            smd_l = SMD()
            smd_l.workflow.position = copy.copy(gene.workflow.position)
            smd_l.workflow.sequence = copy.copy(gene.workflow.sequence)
            ind_l.chromosome.append(smd_l)

        self.crossoverOperator(ind_k, ind_l)
        self.mutantOperator(ind_k)
        self.calculateFitness(ind_k)
        return ind_k


    def crossoverOperator(self, ind_k, ind_l):
        for i in range(self.totalSMDNumber):
            gene_1 = ind_k.chromosome[i]
            gene_2 = ind_l.chromosome[i]
            cpt = random.randint(0, len(gene_1.workflow.position) - 1)
            cPart_1 = []  # 保存第一个个体的执行顺序的从开始到交叉点的片段
            cPart_2 = []  # 保存第二个个体的执行顺序的从开始到交叉点的片段
            # 执行位置交叉
            for j in range(0, cpt):
                gene_1.workflow.position[j], gene_2.workflow.position[j] = gene_2.workflow.position[j], gene_1.workflow.position[j]
                cPart_1.append(gene_1.workflow.sequence[j])
                cPart_2.append(gene_2.workflow.sequence[j])
            # 执行顺序交叉
            for j in range(len(cPart_1)):
                gene_2.workflow.sequence.remove(cPart_1[j])  # 在个体二中移除第一个个体的交叉片段
                gene_1.workflow.sequence.remove(cPart_2[j])  # 在个体一中移除第二个个体的交叉片段
            gene_1.workflow.sequence = cPart_2 + gene_1.workflow.sequence
            gene_2.workflow.sequence = cPart_1 + gene_2.workflow.sequence


    def mutantOperator(self, ind):
        for gene in ind.chromosome:
            rnd_SMD = myRandom.get_0to1_RandomNumber()
            if (rnd_SMD < 1.0 / self.totalSMDNumber):  # 针对每一个基因（SMD）判断是否变异
                for i in range(gene.workflow.position.__len__()):
                    rnd_bit = myRandom.get_0to1_RandomNumber()
                    if (rnd_bit < 1.0 / (gene.workflow.taskNumber)):
                        pos = gene.workflow.position[i]
                        rand = [1, 2, 3, 4, 5, 6, 7]
                        rand.remove(pos)
                        gene.workflow.position[i] = random.choice(rand)

                r = random.randint(1, gene.workflow.sequence.__len__() - 2)  # 随机选择一个变异位置
                formerSetPoint = []
                rearSetPoint = []
                for i in range(0, gene.workflow.sequence.__len__() - 1):  # 从前往后直到所有的前驱任务都被包含在formerSetPoint中
                    formerSetPoint.append(gene.workflow.sequence[i])
                    if set(gene.workflow.taskSet[r].preTaskSet).issubset(set(formerSetPoint)):
                        break
                for j in range(gene.workflow.sequence.__len__() - 1, -1, -1):  # 从后往前直到所有的后继任务都被包含在rearSetPoint中
                    rearSetPoint.append(gene.workflow.sequence[j])
                    if set(gene.workflow.taskSet[r].sucTaskSet).issubset(set(rearSetPoint)):
                        break
                rnd_insert_pt = random.randint(i + 1, j - 1)  # 从i+1到j-1之间随机选一个整数
                gene.workflow.sequence.remove(r)  # 移除变异任务
                gene.workflow.sequence.insert(rnd_insert_pt, r)  # 在随机生成的插入点前插入r


    def updateReferencePoint(self, y_):
        for j in range(self.objectNumber):
            if(self.Z[j] > y_.fitness[j]):
                self.Z[j] = y_.fitness[j]


    def updateNeighborSolutions(self, i, y_):
        for j in self.B[i]:
            y_g_te = self.getTchebycheffValue(j, y_)
            neig_g_te = self.getTchebycheffValue(j, self.population[j])
            if(y_g_te <= neig_g_te):
                self.population[j] = y_


    def update_EP_FromElement(self, EP, ind):  #用新解ind来更新EP
        if EP == []:
            EP.append(copy.deepcopy(ind))
        else:
            i = 0
            while (i < len(EP)):  # 判断ind是否支配EP中的非支配解，若支配，则删除它所支配的解
                if (self.isDominated(ind.fitness, EP[i].fitness) == True):
                    EP.remove(EP[i])
                    i -= 1
                i += 1
            for ep in EP:
                if (self.isDominated(ep.fitness, ind.fitness) == True):
                    return None
            if (self.isExist(ind, EP) == False):
                EP.append(copy.deepcopy(ind))


    def getTchebycheffValue(self, index, ind):  #index是fitness个体的索引，用来获取权重向量
        g_te = []
        for i in range(self.objectNumber):
            temp = self.VT[index][i] * abs(ind.fitness[i] - self.Z[i])
            g_te.append(temp)
        return max(g_te)


    def isExist(self, ind, EP):   #判断个体ind的适应度是否与EP中某个个体的适应度相对，若相等，则返回True
        for ep in EP:
            if ind.fitness == ep.fitness: # 判断两个列表对应元素的值是否相等
                return True
        return False


    def isEP_Dominated_ind(self, ind, EP):   #判断EP中的某个个体是否支配ind，若支配，则返回True
        for ep in EP:
            if self.isDominated(ep.fitness, ind.fitness):
                return True
        return False


    def fast_non_dominated_sort(self, population):
        for p in population:
            p.S_p = []
            p.rank = None
            p.n = 0

        self.F_rank = []
        F1 = []  # 第一个非支配解集前端
        self.F_rank.append(None)
        for p in population:
            for q in population:
                if self.isDominated(p.fitness, q.fitness):
                    p.S_p.append(q)
                elif self.isDominated(q.fitness, p.fitness):
                    p.n += 1
            if (p.n == 0):
                p.rank = 1
                F1.append(p)
        self.F_rank.append(F1)

        i = 1
        while (self.F_rank[i] != []):
            Q = []
            for p in self.F_rank[i]:
                for q in p.S_p:
                    q.n -= 1
                    if (q.n == 0):
                        q.rank = i + 1
                        Q.append(q)

            if(Q != []):
                i += 1
                self.F_rank.append(Q)
            else:
                break


    def isDominated(self, fitness_1, fitness_2):  # 前者是否支配后者
        flag = -1
        for i in range(self.objectNumber):
            if fitness_1[i] < fitness_2[i]:
                flag = 0
            if fitness_1[i] > fitness_2[i]:
                return False
        if flag == 0:
            return True
        else:
            return False




    def calculateFitness(self, ind):
        ind.fitness = []
        time = []
        energy = []
        for gene in ind.chromosome:
            smd = gene  #一个gene就是一个SMD
            self.calculateWorkflowTimeEnergy(smd, smd.workflow)
            time.append(smd.workflow.schedule.T_total)
            energy.append(smd.workflow.schedule.E_total)
        ind.fitness.append(np.average(time))
        ind.fitness.append(np.average(energy))

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
        """获取云中心执行的就绪时间"""
        pre_max = []
        for pre_taskId in task.preTaskSet:
            pre_task = workflow.taskSet[pre_taskId]
            if pre_task.exePosition == 7:  # 前驱任务在云中心执行
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
        schedule.S = {1: [], 2: [], 3: [], 4: [], 5: [], 6: [], 7: []}  # 1-3: 本地核心, 4-6: 边缘节点, 7: 云中心
        schedule.coreTP = {1:[0], 2:[0], 3:[0]}
        schedule.wsTP = [0]
        schedule.MECTP = {}
        for node in self.MEC_nodes:
            schedule.MECTP[node["id"]] = [0]
        schedule.cloudTP = [0]  # 云中心时间点
        schedule.wrTP = [0]
        schedule.T_total = None
        schedule.E_total = 0
        schedule.TimeEnergy = []


    def initializeWorkflowSequence(self, workflow):
        S = []  # 待排序的任务集合
        R = []  # 已排序任务
        T = []
        R.append(workflow.entryTask)
        for task in workflow.taskSet:
            T.append(task.id)
        T.remove(workflow.entryTask)

        while T != []:
            for t in T:
                if set(workflow.taskSet[t].preTaskSet).issubset(set(R)):  #判断t的前驱节点集是否包含在R中
                    if t not in S:
                        S.append(t)
            ti = random.choice(S) #随机从S中选择一个元素
            S.remove(ti)
            T.remove(ti)
            R.append(ti)
        return R




    def calculateInterference(self):
        for i in range(self.M):
            for j in range(self.SeNBSet[i].SMDNumber):
               I_i_j = 0
               for m in range(self.M):
                   if(self.SeNBSet[m] != self.SeNBSet[i]):
                       for k in range(self.SeNBSet[m].SMDNumber):
                           if(self.SeNBSet[m].SMDSet[k].channel == self.SeNBSet[i].SMDSet[j].channel):  # U_m_j and U_l_i have the same channel
                               g_i_m_k = self.getChannelGain(self.SeNBSet[m].SMDSet[k].coordinate, self.SeNBSet[i].coordinate)
                               I_i_j += self.SeNBSet[m].SMDSet[k].pws_i_j * g_i_m_k
               self.SeNBSet[i].SMDSet[j].I_i_j = I_i_j


    def calculateDataTransmissionRate(self):
        self.calculateChannelGain()
        for i in range(self.M):
            for j in range(self.SeNBSet[i].SMDNumber):
                log_v = 1 + (self.SeNBSet[i].SMDSet[j].pws_i_j*self.SeNBSet[i].SMDSet[j].g_i_j) / (self.noisePower + self.SeNBSet[i].SMDSet[j].I_i_j)
                self.SeNBSet[i].SMDSet[j].R_i_j = self.w * math.log(log_v, 2)


    def calculateChannelGain(self):  #calculate G_m_j between SMD U_m_j and SeNB S_m
        for i in range(self.M):
            for j in range(self.SeNBSet[i].SMDNumber):
                self.SeNBSet[i].SMDSet[j].g_i_j = self.getChannelGain(self.SeNBSet[i].SMDSet[j].coordinate, self.SeNBSet[i].coordinate)


    def getChannelGain(self, U_i_j, S_i):  # channel gain= D^(-pl), where D is the distance between U_m_j and S_m, pl=4 is the path loss factor
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
                if (predecessor != ''):
                    predecessor = predecessor.split(',')
                    for pt in predecessor:
                        task.preTaskSet.append(int(pt))
                else:
                    wf.entryTask = int(id)
                task.id = int(id)
                if (successor != ''):
                    successor = successor.split(',')
                    for st in successor:
                        task.sucTaskSet.append(int(st))
                else:
                    wf.exitTask = int(id)
                wf.taskSet.append(task)
        return wf


    def readMECNetwork(self):
        file_SMD_task_cpu = open(self.getCurrentPath() + '\\' + self.taskNumberRange + '\SMD_Task_CPU_Cycles_Number.txt', 'r')
        file_SMD_task_data = open(self.getCurrentPath() + '\\' + self.taskNumberRange + '\SMD_Task_Data_Size.txt', 'r')
        file_SMD_output_task_data = open(self.getCurrentPath() + '\\' + self.taskNumberRange + '\SMD_Task_Output_Data_Size.txt', 'r')

        SeNB_count = -1
        with open(self.getCurrentPath() + '\\' + self.taskNumberRange + '\MEC_Network.txt', 'r') as readFile:
            for line in readFile:
                if(line == '---file end---\n'):
                    break
                elif(line == 'SeNB:\n'):
                    SeNB_count += 1
                    senb = SeNB()     #create SeNB cell
                    if(readFile.readline() == 'Coordinate:\n'):
                        SeNB_crd = readFile.readline()
                        SeNB_crd = SeNB_crd.splitlines()
                        SeNB_crd = SeNB_crd[0].split('  ')
                        senb.coordinate.append(float(SeNB_crd[0]))
                        senb.coordinate.append(float(SeNB_crd[1]))

                        if(readFile.readline() == 'SMD number:\n'):
                            senb.SMDNumber = int(readFile.readline())

                        for line1 in readFile:
                            if (line1 == '---SeNB end---\n'):
                                break
                            elif(line1 == 'SMD:\n'):
                                self.totalSMDNumber += 1
                                smd = SMD()
                                if (readFile.readline() == 'Coordinate:\n'):
                                    SMD_crd = readFile.readline()
                                    SMD_crd = SMD_crd.splitlines()
                                    SMD_crd = SMD_crd[0].split('  ')
                                    smd.coordinate.append(float(SMD_crd[0]))
                                    smd.coordinate.append(float(SMD_crd[1]))

                                if (readFile.readline() == 'Computation capacity:\n'):
                                    SMD_cc = readFile.readline()
                                    SMD_cc = SMD_cc.splitlines()
                                    SMD_cc = SMD_cc[0].split('  ')
                                    smd.coreCC[1] = float(SMD_cc[0])
                                    smd.coreCC[2] = float(SMD_cc[1])
                                    smd.coreCC[3] = float(SMD_cc[2])

                                if (readFile.readline() == 'The number of task:\n'):  #在SeNB（SeNB_count）下得到一个工作流
                                    taskNumber = int(readFile.readline())
                                    SeNB_directory = "SeNB-"+str(SeNB_count)+"\\t"+str(taskNumber)+".txt"
                                    wf_directory = self.getCurrentPath()+"\workflowSet\\"+SeNB_directory
                                    smd.workflow = self.getWorkflow(wf_directory)
                                    smd.workflow.taskNumber = taskNumber
                                    self.codeLength += taskNumber
                                    for task in smd.workflow.taskSet:
                                        task.c_i_j_k = float(file_SMD_task_cpu.readline())    #读取执行任务需要的cpu循环数量
                                        task.d_i_j_k = float(file_SMD_task_data.readline()) * 1024
                                        task.o_i_j_k = float(file_SMD_output_task_data.readline()) * 1024

                                if (readFile.readline() == 'Channel:\n'):
                                    channel = readFile.readline()
                                    smd.channel = int(channel)

                                senb.SMDSet.append(smd)
                    self.SeNBSet.append(senb)
        file_SMD_task_data.close()
        file_SMD_task_cpu.close()



    def getCurrentPath(self):
        return os.path.dirname(os.path.realpath(__file__))





class Individual:
    def __init__(self):
        self.chromosome = []      #基因位是SMD类型
        self.fitness = []
        self.isFeasible = True    #判断该个体是否合法
        self.temp_fitness = None  #临时适应度，计算拥挤距离的时候，按每个目标值来对类列表进行升序排序
        self.distance = 0.0
        self.rank = None
        self.S_p = []  #种群中此个体支配的个体集合
        self.n = 0  #种群中支配此个体的个数

class SeNB:
    def __init__(self):
        self.coordinate = []  # SeNB的位置坐标
        self.SMDNumber = 0    # SMD设备数量
        self.SMDSet = []      # 该SeNB覆盖的SMD设备集合

class SMD:
    def __init__(self):
        self.coordinate = []          # SMD设备的位置坐标
        self.workflow = Workflow()    # SMD设备的工作流
        self.channel = None           # 获取的信道索引
        self.g_i_j = None             # SMD与SeNB之间的信道增益
        self.R_i_j = None             # SMD的数据传输速率
        self.I_i_j = None             # SMD受到的干扰
        # SMD被建模为一个4元组
        self.coreCC = {1:None, 2:None, 3:None}  # 核心的计算能力

        self.pcc_i_j = {1:4, 2:2, 3:1}  # 三个核心在最大工作频率下的功耗
        self.pws_i_j = 0.5  # SMD的发送数据功率（瓦）
        self.pwr_i_j = 0.1  # SMD的接收数据功率（瓦）

class Workflow:
    def __init__(self):
        self.entryTask = None      # 开始任务
        self.exitTask = None       # 结束任务
        self.position = []         # 执行位置
        self.sequence = []         # 执行顺序
        self.taskNumber = None     # 任务数量
        self.taskSet = []          # 任务集合（列表索引值就是任务的id值）
        self.schedule = Schedule() # 调度信息

class Schedule:
    def __init__(self):
        self.taskSet = {}          # 任务集合
        self.S = {1: [], 2: [], 3: [], 4: [], 5: [], 6: [], 7: []}  # 执行单元：1-3本地核心，4-6边缘节点，7云中心
        self.coreTP = {1:[0], 2:[0], 3:[0]}  # 键为核心编号，元素表示该核心上的当前时间点
        self.wsTP = [0]  # 无线发送信道上的当前时间点

        self.MECTP = {}  # MEC节点时间点，将在运行时初始化
        self.cloudTP = [0]  # 云中心时间点

        self.wrTP = [0]  # 无线接收信道上的当前时间点
        self.T_total = None  # 总时间
        self.E_total = 0     # 总能耗
        self.TimeEnergy = [] # 时间能耗记录

class Task:
    def __init__(self):
        self.id = None             # 任务ID
        self.islocal = None        # 表示任务是在本地执行还是在云端执行
        self.preTaskSet = []       # 前驱任务集合（元素为Task类）
        self.sucTaskSet = []       # 后继任务集合（元素为Task类）
        self.exePosition = None    # 任务的执行位置（即[1,2,3,4]）
        self.actualFre = 1         # 实际频率缩放因子
        self.c_i_j_k = None        # 执行任务所需的CPU周期数
        self.d_i_j_k = None        # 任务的数据大小
        self.o_i_j_k = None        # 任务的输出数据大小

        self.RT_i_l = None         # 任务vi在本地核心上的就绪时间
        self.RT_i_ws = None        # 任务vi在无线发送信道上的就绪时间
        self.RT_i_c = None         # 任务vi在[10,20]服务器上的就绪时间
        self.RT_i_wr = None        # 云端传输回任务vi结果的就绪时间
        self.RT_i_cloud = None  # 云中心就绪时间

        self.ST_i_l = None         # 任务vi在本地核心上的开始时间
        self.ST_i_ws = None        # 任务vi在无线发送信道上的开始时间
        self.ST_i_c = None         # 任务vi在[10,20]服务器上的开始时间
        self.ST_i_wr = None        # 云端传输回任务vi结果的开始时间
        self.ST_i_cloud = None  # 云中心开始时间

        self.FT_i_l = None         # 任务vj在本地核心上的完成时间
        self.FT_i_ws = None        # 任务vj在无线发送信道上的完成时间
        self.FT_i_c = None         # 任务vj在[10,20]服务器上的完成时间
        self.FT_i_wr = None        # 任务vj在无线接收信道上的完成时间
        self.energy = 0            # 能耗
        self.FT_i_cloud = None  # 云中心完成时间