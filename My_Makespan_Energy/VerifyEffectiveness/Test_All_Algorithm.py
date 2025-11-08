import os,sys
sys.path.append("./") # 修改 VS自定义包要改路径 搞不懂这个文件路径
from My_Makespan_Energy.VerifyEffectiveness import MOEAD, MOEAD_1, MOEAD_ADW, MOEAD_ADW1, MOEAD_ADW11
from My_Makespan_Energy.VerifyEffectiveness import AR_MOEA, MOEAD_AWS

import copy
import time
from operator import itemgetter
import matplotlib.pyplot as plt
import numpy as np
import xlrd
import xlwt # 修改 备注 xlwt库可以生成 .xls文件




class Test_All_Algorithm:
    # def __init__(self):
    #     algorithmName_list = ["NSGA2", "DVFS", "MOEAD", "MOEAD_DVFS"]
    #     for TN in range(10, 101, 10):
    #         self.getReferPF(TN, algorithmName_list)

    def getIGD(self, algorithmName_list, runTime, taskNumberRange):
        readAlgorithmPath = self.getCurrentPath() + "\ExperimentResult\\" + taskNumberRange + "\\"
        readReferPFPath = self.getCurrentPath() + "\ExperimentResult\\" + taskNumberRange + "\\referPF.xls"
        IGDFilePath = self.getCurrentPath() + "\ExperimentResult\\" + taskNumberRange + "\\"+"IGD.xls"
        data = xlrd.open_workbook(readReferPFPath)
        table = data.sheet_by_name('total')
        PF_ref = table._cell_values

        f = xlwt.Workbook(IGDFilePath)
        IGD_mean_std_total_list = []
        for alg in algorithmName_list:
            sheet = f.add_sheet(alg)
            filePath = readAlgorithmPath + alg + ".xls"
            data = xlrd.open_workbook(filePath)
            if alg != 'DVFS':
                IGDValue_list = []
                for i in range(0,runTime):
                    table = data.sheet_by_name(str(i+1))
                    PF_know = table._cell_values
                    PF_know.pop() # 删除最后一行的计算时间
                    IGDValue = self.getIGDValue(PF_ref, PF_know)
                    IGDValue_list.append(IGDValue)
                    sheet.write(i, 0, IGDValue)
                mean = float(np.mean(IGDValue_list))
                std = float(np.std(IGDValue_list, ddof=1))
                sheet.write(i+1, 0, mean)
                sheet.write(i+1, 1, std)
                temp = []
                temp.append(mean)
                temp.append(std)
                IGD_mean_std_total_list.append(temp)
            else:
                table = data.sheet_by_name('1')
                PF_know = table._cell_values
                PF_know.pop()  # 删除最后一行的计算时间
                IGDValue = self.getIGDValue(PF_ref, PF_know)
                sheet.write(0, 0, IGDValue)
                temp = []
                temp.append(IGDValue)
                temp.append(0)
                IGD_mean_std_total_list.append(temp)

        sheet_IGD_total = f.add_sheet('IGD-total')
        i=1
        for IGD_mean_std in IGD_mean_std_total_list:
            sheet_IGD_total.write(i, 0, round(IGD_mean_std[0], 4))
            sheet_IGD_total.write(i, 1, round(IGD_mean_std[1], 4))
            i+=1
        f.save(IGDFilePath)


    def getIGDValue(self, PF_ref, PF_know):
        sum = []
        for v in PF_ref:
            distance = self.d_v_PFSet(v, PF_know)
            sum.append(distance)
        return np.average(sum)


    def getGD(self, algorithmName_list, runTime, taskNumberRange):
        readAlgorithmPath = self.getCurrentPath() + "\ExperimentResult\\" + taskNumberRange + "\\"
        readReferPFPath = self.getCurrentPath() + "\ExperimentResult\\" + taskNumberRange + "\\referPF.xls"
        GDFilePath = self.getCurrentPath() + "\ExperimentResult\\" + taskNumberRange + "\\" + "GD.xls"
        data = xlrd.open_workbook(readReferPFPath)
        table = data.sheet_by_name('total')
        PF_ref = table._cell_values

        f = xlwt.Workbook(GDFilePath)
        GD_mean_std_total_list = []

        for alg in algorithmName_list:
            sheet = f.add_sheet(alg)
            filePath = readAlgorithmPath + alg + ".xls"
            data = xlrd.open_workbook(filePath)
            if alg != 'DVFS':
                GDValue_list = []
                for i in range(0, runTime):
                    table = data.sheet_by_name(str(i + 1))
                    PF_know = table._cell_values
                    PF_know.pop()  # 删除最后一行的计算时间
                    GDValue = self.getGDValue(PF_ref, PF_know)
                    GDValue_list.append(GDValue)
                    sheet.write(i, 0, GDValue)
                mean = float(np.mean(GDValue_list))
                std = float(np.std(GDValue_list, ddof=1))
                sheet.write(i + 1, 0, mean)
                sheet.write(i + 1, 1, std)
                temp = []
                temp.append(mean)
                temp.append(std)
                GD_mean_std_total_list.append(temp)

            else:
                table = data.sheet_by_name('1')
                PF_know = table._cell_values
                PF_know.pop()  # 删除最后一行的计算时间
                GDValue = self.getGDValue(PF_ref, PF_know)
                sheet.write(0, 0, GDValue)
                temp = []
                temp.append(GDValue)
                temp.append(0)
                GD_mean_std_total_list.append(temp)
        sheet_GD_total = f.add_sheet('GD-total')
        i = 1
        for GD_mean_std in GD_mean_std_total_list:
            sheet_GD_total.write(i, 0, round(GD_mean_std[0], 4))
            sheet_GD_total.write(i, 1, round(GD_mean_std[1], 4))
            i += 1
        f.save(GDFilePath)


    def getGDValue(self, PF_ref, PF_know):
        sum = []
        for v in PF_know:
            distance = self.d_v_PFSet(v, PF_ref)
            sum.append(distance)
        return np.sqrt(np.average(sum))

   
    def d_v_PFSet(self, v, PFSet):  # 求v和PFSet中最近的距离
        dList = []
        for pf in PFSet:
            distance = self.getDistance(v, pf)
            dList.append(distance)
        return min(dList)


    def getDistance(self, point1, point2):
        return np.sqrt(np.sum(np.square([point1[i] - point2[i] for i in range(2)])))


    def update_EP_History(self, EP_Current, EP_History):  # 用当前运行后的非支配解集EP_Current来更新历史非支配解集EP_History
        if (EP_History == []):
            for epc in EP_Current:
                EP_History.append(copy.copy(epc))
        else:
            for epc in EP_Current:
                if (self.isExist(epc, EP_History) == False):  # 先判断ep是否在EP_History中，若不在，则返回False。
                    if (self.isEP_Dominated_ind(EP_History, epc) == False):  # 然后再判断EP_History是否支配ep
                        i = 0
                        while (i < EP_History.__len__()):  # 判断ep是否支配EP中的非支配解，若支配，则删除它所支配的解
                            if (self.isDominated(epc, EP_History[i]) == True):
                                EP_History.remove(EP_History[i])
                                i -= 1
                            i += 1
                        EP_History.append(copy.copy(epc))


    def isExist(self, ep, EP_History):   #判断ep是否与EP中某个支配解相对，若相等，则返回True
        for eph in EP_History:
            if ep == eph: # 判断两个列对应元素的值是否相等
                return True
        return False


    def isEP_Dominated_ind(self, EP_History, ep):   #判断EP中的某个非支配解是否支配ep，若支配，则返回True
        for eph in EP_History:
            if self.isDominated(eph, ep):
                return True
        return False


    def isDominated(self, fitness_1, fitness_2):  # 前者是否支配后者
        flag = -1
        for i in range(2):
            if fitness_1[i] < fitness_2[i]:
                flag = 0
            if fitness_1[i] > fitness_2[i]:
                return False
        if flag == 0:
            return True
        else:
            return False


    def plotEP(self, EP):
        x = []
        y = []
        for ep in EP:
            x.append(ep[0])
            y.append(ep[1])
        plt.scatter(x, y, marker='o')
        plt.grid(True)
        plt.xlabel('Makespan (s)')
        plt.ylabel('Energy Consumption (w)')
        plt.title('Non-dominated Solution')
        plt.show()




    def getReferParetoFront(self, algorithmName_list, taskNumberRange):
        readPath = self.getCurrentPath() + "\ExperimentResult\\" + taskNumberRange + "\\"
        writePath = self.getCurrentPath() + "\ExperimentResult\\" + taskNumberRange + "\\referPF.xls"
        referPF = []
        for alg in algorithmName_list:
            filePath = readPath + alg+".xls"
            data = xlrd.open_workbook(filePath)
            table = data.sheet_by_name('total')
            currentPF = table._cell_values
            currentPF.pop() #删除最后一行的运行时间
            self.update_EP_History(currentPF, referPF)

        referPF = sorted(referPF, key=itemgetter(0))
        f = xlwt.Workbook(writePath)
        sheet = f.add_sheet('total')
        for i in range(len(referPF)):
            sheet.write(i, 0, referPF[i][0])
            sheet.write(i, 1, referPF[i][1])
        f.save(writePath)


    # def plot_All_ACT_AEC_BoxGraph(self, algorithmName_list, taskNumberRange):
    #     completionTime = []
    #     energyConsumption = []
    #     for algorithmName in algorithmName_list:
    #         ctList, ecList = self.getEPSet_FT_EC(algorithmName, taskNumberRange)
    #         completionTime.append(ctList)
    #         energyConsumption.append(ecList)
    #
    #     plt.close()
    #     font = {'family': 'Times New Roman', 'weight': 'normal', 'size': 10}
    #     plt.xlabel('Algorithm', font)
    #     plt.ylabel('ACT (Sec.)', font)
    #
    #     # 动态创建显示名称映射
    #     display_names = {
    #         'AR_MOEA': 'AR-MOEA',
    #         'MOEAD': 'MOEA/D',
    #         "MOEAD_AWS": "MOEAD_AWS",
    #         'MOEAD_ADW': 'MOEAD_ADW',
    #
    #         'MOEAD_ADW11': 'MOEAD-ADW11'
    #     }
    #
    #     # 根据传入的算法名称生成对应的显示名称
    #     display_name_list = [display_names.get(name, name) for name in algorithmName_list]
    #
    #     plt.boxplot(completionTime,
    #                 tick_labels=display_name_list,
    #                 flierprops={'marker': 'o', 'markerfacecolor': 'red', 'markersize': 3},
    #                 meanprops={'marker': 'D', 'markerfacecolor': 'indianred', 'markersize': 4}, )
    #
    #     plt.xticks(rotation=18)
    #     filename = self.getCurrentPath() + "\\ExperimentResult\\" + taskNumberRange + '\\' + 'PFBox_ACT_All.pdf'
    #     plt.savefig(filename, bbox_inches='tight')
    #
    #     plt.close()
    #
    #     plt.boxplot(energyConsumption,
    #                 tick_labels=display_name_list,
    #                 flierprops={'marker': 'o', 'markerfacecolor': 'red', 'markersize': 3},
    #                 meanprops={'marker': 'D', 'markerfacecolor': 'indianred', 'markersize': 4}, )
    #
    #     plt.xticks(rotation=18)
    #     plt.xlabel('Algorithm', font)
    #     plt.ylabel('AEC (J)', font)
    #
    #     filename = self.getCurrentPath() + "\\ExperimentResult\\" + taskNumberRange + '\\' + 'PFBox_AEC_All.pdf'
    #     plt.savefig(filename, bbox_inches='tight')
    #
    #     plt.close()

    def plot_All_ACT_AEC_BoxGraph(self, algorithmName_list, taskNumberRange):
        import numpy as np
        import matplotlib.pyplot as plt

        completionTime = []
        energyConsumption = []
        for algorithmName in algorithmName_list:
            ctList, ecList = self.getEPSet_FT_EC(algorithmName, taskNumberRange)
            completionTime.append(ctList)
            energyConsumption.append(ecList)

        # ======== 异常值剔除函数 ========
        def trim_outliers(data, lower=5, upper=95):
            if len(data) == 0:
                return data
            low, high = np.percentile(data, [lower, upper])
            return [x for x in data if low <= x <= high]

        completionTime = [trim_outliers(lst) for lst in completionTime]
        energyConsumption = [trim_outliers(lst) for lst in energyConsumption]

        # ======== 显示名称映射 ========
        font = {'family': 'Times New Roman', 'weight': 'normal', 'size': 10}
        display_names = {
            'AR_MOEA': 'AR-MOEA',
            'MOEAD': 'MOEA/D',
            'MOEAD_AWS': 'MOEAD_AWS',
            'MOEAD_ADW': 'MOEAD_ADW',
            'MOEAD_ADW11': 'MOEAD-ADW11'
        }
        display_name_list = [display_names.get(name, name) for name in algorithmName_list]

        # ======== 自定义颜色（可根据算法数量调整） ========
        colors = ['#FF9999', '#66B2FF', '#99FF99', '#FFCC99', '#C2C2F0', '#FFD700']

        # ==================== 绘制 ACT ====================
        plt.close()
        fig, ax = plt.subplots()
        bp1 = ax.boxplot(
            completionTime,
            tick_labels=display_name_list,
            showfliers=False,
            patch_artist=True,
            flierprops={'marker': 'o', 'markerfacecolor': 'red', 'markersize': 3},
            meanprops={'marker': 'D', 'markerfacecolor': 'indianred', 'markersize': 4}
        )

        for patch, color in zip(bp1['boxes'], colors[:len(completionTime)]):
            patch.set_facecolor(color)

        ax.set_xlabel('Algorithm', font)
        ax.set_ylabel('ACT (Sec.)', font)
        plt.xticks(rotation=18)
        plt.tight_layout()
        plt.savefig(self.getCurrentPath() + f"\\ExperimentResult\\{taskNumberRange}\\PFBox_ACT_All.pdf",
                    bbox_inches='tight')
        plt.close()

        # ==================== 绘制 AEC ====================
        plt.close()
        fig, ax = plt.subplots()
        bp2 = ax.boxplot(
            energyConsumption,
            tick_labels=display_name_list,
            showfliers=False,
            patch_artist=True,
            flierprops={'marker': 'o', 'markerfacecolor': 'red', 'markersize': 3},
            meanprops={'marker': 'D', 'markerfacecolor': 'indianred', 'markersize': 4}
        )

        for patch, color in zip(bp2['boxes'], colors[:len(energyConsumption)]):
            patch.set_facecolor(color)

        ax.set_xlabel('Algorithm', font)
        ax.set_ylabel('AEC (J)', font)
        plt.xticks(rotation=18)
        plt.tight_layout()
        plt.savefig(self.getCurrentPath() + f"\\ExperimentResult\\{taskNumberRange}\\PFBox_AEC_All.pdf",
                    bbox_inches='tight')
        plt.close()


    def writeEPToExcelFile(self, f, EP, sheetName, computationTime):
        newEP = sorted(EP, key=itemgetter(0))
        sheet = f.add_sheet(sheetName)
        for i in range(len(newEP)):
            sheet.write(i, 0, newEP[i][0])
            sheet.write(i, 1, newEP[i][1])
        sheet.write(i+1, 0, computationTime)



    def DVFS_writeEPToFile(self, algorithmName, EP, taskNumber, runTime):
        main_path = self.getCurrentPath()+"\ExperimentResult\\"
        filename = algorithmName+"_"+str(taskNumber)+"_"+str(runTime)+".txt"
        with open(main_path+filename, "w") as writeFile:
            for ep in EP:
                writeFile.write(str(ep[0])+"    "+str(ep[1])+"\n")


    def writeEPIndividualToFile(self, EP_ind, algorithmName, popSize, maxGen, run_time):
        main_path = self.getCurrentPath()+"\ExperimentResult\\"
        filename = algorithmName + "_Ind_" + str(popSize) + "_" + str(maxGen) + "_" + str(run_time) + ".txt"
        with open(main_path+filename, "w") as writeFile:
            for epi in EP_ind:
                for gene in epi.chromosome:
                    for pos in gene.workflow.position:
                        writeFile.write(str(pos)+", ")
                    writeFile.write("\n")
                writeFile.write("\n\n\n")


    def getCurrentPath(self):
        return os.path.dirname(os.path.realpath(__file__))


    def getProjectPath(self):
        cur_path = os.path.dirname(os.path.realpath(__file__))
        return os.path.join(os.path.dirname(cur_path))


    def getreferPFFromFile(self, taskNumber):
        filename = self.getProjectPath()+'\My_Makespan_Energy\ExperimentResult\\'+'referPF.xls'
        data = xlrd.open_workbook(filename)
        table = data.sheet_by_name('total')
        PF = table._cell_values
        PF = np.array(PF)
        return (PF[:, 0], PF[:, 1])


    def getParetoFront(self, algorithmName, taskNumberRange):
        filename = self.getCurrentPath()+'\ExperimentResult\\'+taskNumberRange+'\\'+algorithmName+'.xls'
        data = xlrd.open_workbook(filename)
        table = data.sheet_by_name('total')
        PF = table._cell_values
        PF.pop()
        PF = np.array(PF)
        return (PF[:,0], PF[:,1])


    def plotParetoFront(self, algorithmName_list, taskNumberRange):
        XY = []
        FG = []

        for algorithmName in algorithmName_list:
            (x, y) = self.getParetoFront(algorithmName, taskNumberRange)
            XY.append((x, y))

        # 适合空心的标记形状
        Marker = ['o', 's', 'D', '^', '+', 'p', '*']  # 圆圈、方形、菱形、上三角、加号、五角形、星形
        Color = ['purple', 'green', 'blue', 'magenta', 'red', 'orange', 'cyan']

        # 创建算法名称到显示标签的映射
        label_map = {
            "MOEAD_ADW11": "MOEA/D-ADW11",
            "MOEAD_ADW1": "MOEA/D-ADW1",
            "MOEAD_1": "MOEA/D-1",

            'AR_MOEA': 'AR-MOEA',
            'MOEAD': 'MOEA/D',
            "MOEAD_AWS": "MOEAD_AWS",
            'MOEAD_ADW': 'MOEAD-ADW'

        }

        for i in range(len(XY)):
            # 获取显示标签
            display_label = label_map.get(algorithmName_list[i], algorithmName_list[i])

            # 绘制空心标记
            fg, = plt.plot(XY[i][0], XY[i][1],
                           marker=Marker[i],
                           markersize=5,  # 稍微增大以便空心效果更明显
                           color=Color[i],
                           # fillstyle='none',  # 关键参数：设置为空心
                           # markeredgewidth=1,  # 边缘线宽
                           linestyle='',
                           linewidth=2,
                           label=display_label)
            FG.append(fg)

        font = {'size': 13}
        plt.legend(handles=FG, prop=font, loc='center right')
        plt.xlabel('ACT (Sec.)', font)
        plt.ylabel('AEC (J)', font)
        # plt.grid(True, alpha=0.3)  # 添加网格线便于观察

        fig = plt.gcf()
        fig.set_size_inches(10, 8)  #
        filename = self.getCurrentPath() + "\ExperimentResult\\" + taskNumberRange + '\Effectiveness_PFcurve.pdf'
        fig.savefig(filename, bbox_inches='tight', dpi=300)
        fig.clear()


    def getEPSet_FT_EC(self, algorithmName, taskNumberRange):
        filePath = self.getCurrentPath()+'\ExperimentResult\\'+taskNumberRange+'\\'+algorithmName+'.xls'
        data = xlrd.open_workbook(filePath)
        table = data.sheet_by_name('total')
        PF = table._cell_values
        PF.pop()
        PF = np.array(PF)
        return PF[:, 0], PF[:, 1]


    def plotPFSetBoxGraph(self, algorithmName_list, taskNumberRange):
        completionTime = []
        energyConsumption = []
        for algorithmName in algorithmName_list:
            ctList, ecList = self.getEPSet_FT_EC(algorithmName, taskNumberRange)
            completionTime.append(ctList)
            energyConsumption.append(ecList)

        plt.close()
        font={'family': 'Times New Roman', 'weight': 'normal', 'size': 12}
        plt.xlabel('Algorithm', font)
        plt.ylabel('Completion Time (S)', font)
        plt.boxplot(completionTime, labels=algorithmName_list)
        filename = self.getCurrentPath() + "\ExperimentResult\\" +taskNumberRange+ '\PFSetBox_CT'
        # 修改 plt.savefig(filename, figsize=(1, 1), dpi=800)
        plt.savefig(filename, dpi=800)

        plt.close()
        plt.boxplot(energyConsumption, labels=algorithmName_list)
        plt.xlabel('Algorithm', font)
        plt.ylabel('Energy Consumption (J)', font)
        filename = self.getCurrentPath() + "\ExperimentResult\\" +taskNumberRange+ '\PFSetBox_EC'
        # 修改 plt.savefig(filename, figsize=(1, 1), dpi=800)
        plt.savefig(filename, dpi=800)
        plt.close()

    def MOEAD_Run(self, popSize, maxGen, T, runTime, taskNumberRange):
        print("*** MOEAD (Run " + taskNumberRange + ' ' + str(runTime) + " time) ***")
        EP_History = []
        time_list = []
        filename = self.getCurrentPath() + "\ExperimentResult\\" + taskNumberRange + "\\MOEAD.xls"
        f = xlwt.Workbook(filename)
        for I in range(1, runTime + 1):
            print("The " + str(I) + "-th time")
            startTime = time.time()
            moead = MOEAD.MOEAD(popSize, maxGen, T, taskNumberRange)
            EP_Current = moead.run()
            endTime = time.time()
            CT = endTime - startTime
            print("Computation time: ", CT)
            self.writeEPToExcelFile(f, EP_Current, str(I), CT)
            time_list.append(CT)
            self.update_EP_History(EP_Current, EP_History)
        print("ACT: ", np.average(time_list))
        self.writeEPToExcelFile(f, EP_History, 'total', np.average(time_list))
        f.save(filename)

    def MOEAD_1_Run(self, popSize, maxGen, T, runTime, taskNumberRange):
        print("*** MOEAD_1 (Run " +taskNumberRange+' '+ str(runTime) + " time) ***")
        EP_History = []
        time_list = []
        filename = self.getCurrentPath()+"\ExperimentResult\\"+taskNumberRange+"\\MOEAD_1.xls"
        f = xlwt.Workbook(filename)
        for I in range(1, runTime + 1):
            print("The " + str(I) + "-th time")
            startTime = time.time()
            moead = MOEAD_1.MOEAD(popSize, maxGen, T, taskNumberRange)
            EP_Current = moead.run()
            endTime = time.time()
            CT = endTime - startTime
            print("Computation time: ", CT)
            self.writeEPToExcelFile(f, EP_Current, str(I), CT)
            time_list.append(CT)
            self.update_EP_History(EP_Current, EP_History)
        print("ACT: ", np.average(time_list))
        self.writeEPToExcelFile(f, EP_History, 'total', np.average(time_list))
        f.save(filename)


    def MOEAD_ADW11_Run(self, popSize, maxGen, T, runTime, taskNumberRange):
        print("*** MOEAD_ADW11(Run " +taskNumberRange+' '+ str(runTime) + " time) ***")
        EP_History = []
        time_list = []
        filename = self.getCurrentPath()+"\ExperimentResult\\"+taskNumberRange+"\\MOEAD_ADW11.xls"
        f = xlwt.Workbook(filename)
        for I in range(1, runTime + 1):
            print("The " + str(I) + "-th time")
            startTime = time.time()
            moead = MOEAD_ADW11.ARMOEA(popSize, maxGen, T, taskNumberRange)
            EP_Current = moead.run()
            endTime = time.time()
            CT = endTime - startTime
            print("Computation time: ", CT)
            self.writeEPToExcelFile(f, EP_Current, str(I), CT)
            time_list.append(CT)
            self.update_EP_History(EP_Current, EP_History)
        print("ACT: ", np.average(time_list))
        self.writeEPToExcelFile(f, EP_History, 'total', np.average(time_list))
        f.save(filename)

    def MOEAD_ADW_Run(self, popSize, maxGen, T, runTime, taskNumberRange):
        print("*** MOEAD_ADW (Run " +taskNumberRange+' '+ str(runTime) + " time) ***")
        EP_History = []
        time_list = []
        # 构建Excel文件路径字符串，用于存储MOEAD_ADW算法的实验结果
        filename = self.getCurrentPath()+"\ExperimentResult\\"+taskNumberRange+"\\MOEAD_ADW.xls"
        f = xlwt.Workbook(filename) # 使用xlwt库的Workbook类创建一个Excel工作簿对象，上面是路径
        for I in range(1, runTime + 1):
            print("The " + str(I) + "-th time")
            startTime = time.time()
            moead_svfs = MOEAD_ADW.MOEAD(popSize, maxGen, T, taskNumberRange)
            EP_Current = moead_svfs.run()
            endTime = time.time()
            CT = endTime - startTime
            print("Computation time: ", CT)
            self.writeEPToExcelFile(f, EP_Current, str(I), CT)
            time_list.append(CT)
            self.update_EP_History(EP_Current, EP_History)
        print("ACT: ", np.average(time_list))
        self.writeEPToExcelFile(f, EP_History, 'total', np.average(time_list))
        f.save(filename)


    def MOEAD_ADW1_Run(self, popSize, maxGen, T, runTime, taskNumberRange):
        print("*** MOEAD_ADW1 (Run " +taskNumberRange+' '+ str(runTime) + " time) ***")
        EP_History = []
        time_list = []
        filename = self.getCurrentPath()+"\ExperimentResult\\"+taskNumberRange+"\\MOEAD_ADW1.xls"
        f = xlwt.Workbook(filename)
        for I in range(1, runTime + 1):
            print("The " + str(I) + "-th time")
            startTime = time.time()
            moead = MOEAD_ADW1.MOEAD(popSize, maxGen, T, taskNumberRange)
            EP_Current = moead.run()
            endTime = time.time()
            CT = endTime - startTime
            print("Computation time: ", CT)
            self.writeEPToExcelFile(f, EP_Current, str(I), CT)
            time_list.append(CT)
            self.update_EP_History(EP_Current, EP_History)
        print("ACT: ", np.average(time_list))
        self.writeEPToExcelFile(f, EP_History, 'total', np.average(time_list))
        f.save(filename)

    def AR_MOEA_Run(self, popSize, maxGen, T, runTime, taskNumberRange):
        print("*** AR_MOEA(Run " +taskNumberRange+' '+ str(runTime) + " time) ***")
        EP_History = []
        time_list = []
        filename = self.getCurrentPath()+"\ExperimentResult\\"+taskNumberRange+"\\AR_MOEA.xls"
        f = xlwt.Workbook(filename)
        for I in range(1, runTime + 1):
            print("The " + str(I) + "-th time")
            startTime = time.time()
            armoea = AR_MOEA.ARMOEA(popSize, maxGen, T, taskNumberRange)
            EP_Current = armoea.run()
            endTime = time.time()
            CT = endTime - startTime
            print("Computation time: ", CT)
            self.writeEPToExcelFile(f, EP_Current, str(I), CT)
            time_list.append(CT)
            self.update_EP_History(EP_Current, EP_History)
        print("ACT: ", np.average(time_list))
        self.writeEPToExcelFile(f, EP_History, 'total', np.average(time_list))
        f.save(filename)

    def MOEAD_AWS_Run(self, popSize, maxGen, T, runTime, taskNumberRange):
        print("*** MOEAD_AWS(Run " +taskNumberRange+' '+ str(runTime) + " time) ***")
        EP_History = []
        time_list = []
        filename = self.getCurrentPath()+"\ExperimentResult\\"+taskNumberRange+"\\MOEAD_AWS.xls"
        f = xlwt.Workbook(filename)
        for I in range(1, runTime + 1):
            print("The " + str(I) + "-th time")
            startTime = time.time()
            aws = MOEAD_AWS.MOEAD(popSize, maxGen, T, taskNumberRange)
            EP_Current = aws.run()
            endTime = time.time()
            CT = endTime - startTime
            print("Computation time: ", CT)
            self.writeEPToExcelFile(f, EP_Current, str(I), CT)
            time_list.append(CT)
            self.update_EP_History(EP_Current, EP_History)
        print("ACT: ", np.average(time_list))
        self.writeEPToExcelFile(f, EP_History, 'total', np.average(time_list))
        f.save(filename)


if __name__=="__main__":
    popSize = 100  # 从100改为20
    maxGen = 100    # 从100改为5
    runTime = 20   # 从20改为1
    # popSize = 20
    # maxGen = 5
    # runTime = 1
    pc = 0.8   # 交叉概率
    pm_SMD = 0.03  #SMD变异概率
    pm_bit = 0.01  # 基因位变异概率  Bit Mutation Probability
    EP_Current_list = []
    EP_History = []

    # taskNumberRangeList = [ '[20,30]', '[30,40]', '[40,60]' ]

    # taskNumberRangeList = ['[10,20]']   # Td_Range = [4,7]
    # taskNumberRangeList = ['[20,30]']   # Td_Range = [6.5,10]
    # taskNumberRangeList = ['[30,40]']   # Td_Range = [9,12.5]
    taskNumberRangeList = ['[40,60]']  # Td_Range = [5,8]



    Test = Test_All_Algorithm()
    for taskNumberRange in taskNumberRangeList:


        # Test.MOEAD_Run(popSize, maxGen, 10, runTime, taskNumberRange)
        # Test.AR_MOEA_Run(popSize, maxGen,10, runTime, taskNumberRange)
        # Test.MOEAD_AWS_Run(popSize, maxGen, 10, runTime, taskNumberRange)

        # Test.MOEAD_1_Run(popSize, maxGen, 10, runTime, taskNumberRange)

        # Test.MOEAD_ADW_Run(popSize, maxGen, 10, runTime, taskNumberRange)
        # Test.MOEAD_ADW1_Run(popSize, maxGen, 10, runTime, taskNumberRange)
        # Test.MOEAD_ADW11_Run(popSize, maxGen, 10, runTime, taskNumberRange)



        # Test.getReferParetoFront([ 'AR_MOEA', 'MOEAD', 'MOEAD_AWS', 'MOEAD_ADW' ], taskNumberRange)
        # Test.getReferParetoFront(['AR_MOEA', 'MOEAD', 'MOEAD_AWS', 'MOEAD_ADW', 'MOEAD_ADW11' ], taskNumberRange)
        Test.getReferParetoFront(['MOEAD', 'MOEAD_1', 'MOEAD_ADW' ], taskNumberRange)
        # Test.getReferParetoFront([ 'AR_MOEA', 'MOEAD_ADW11' ], taskNumberRange)


        # Test.plotParetoFront([ 'AR_MOEA', 'MOEAD', 'MOEAD_AWS', 'MOEAD_ADW' ], taskNumberRange)
        # Test.plotParetoFront([ 'AR_MOEA', 'MOEAD', 'MOEAD_AWS', 'MOEAD_ADW', 'MOEAD_ADW11' ], taskNumberRange)
        # Test.plotParetoFront([ 'MOEAD', 'MOEAD_1', 'MOEAD_ADW'], taskNumberRange)
        # Test.plotParetoFront([ 'AR_MOEA', 'MOEAD_ADW11' ], taskNumberRange)


        
        Test.plot_All_ACT_AEC_BoxGraph([ 'AR_MOEA', 'MOEAD', 'MOEAD_AWS', 'MOEAD_ADW' ], taskNumberRange)
        # Test.plot_All_ACT_AEC_BoxGraph(['AR_MOEA', 'MOEAD', 'MOEAD_AWS', 'MOEAD_ADW', 'MOEAD_ADW11' ], taskNumberRange)

        # Test.getIGD([ 'AR_MOEA', 'MOEAD', 'MOEAD_AWS', 'MOEAD_ADW' ], runTime, taskNumberRange)
        # Test.getGD([ 'AR_MOEA', 'MOEAD', 'MOEAD_AWS', 'MOEAD_ADW' ], runTime, taskNumberRange)


