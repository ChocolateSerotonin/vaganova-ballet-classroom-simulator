import mesa
import numpy as np
from agents import StudentAgent, Teacher


class ClassroomModel(mesa.Model):
    def __init__(self, 
                 n_initial=20,       # 初始学生数
                 p_attend=0.8,       # 出勤概率
                 # 学习参数
                 alpha=10, gamma=15, beta=2, sigma=3,
                 # 增长参数
                 eta_B=2, eta_E_low=0.02, eta_E_high=0.08, k=5,
                 B_max=100, delta_M=2,
                 # 新学生流入
                 new_student_rate=0.05,
                 seed=None):
        
        super().__init__(seed=seed)
        
        # 存参数
        self.alpha = alpha
        self.gamma = gamma
        self.beta = beta
        self.sigma = sigma
        self.eta_B = eta_B
        self.eta_E_low = eta_E_low
        self.eta_E_high = eta_E_high
        self.k = k
        self.B_max = B_max
        self.delta_M = delta_M
        self.p_attend = p_attend
        self.new_student_rate = new_student_rate
        
        # 调度器
        self.schedule = mesa.time.BaseScheduler(self)
        
        # 创建教师
        self.teacher = Teacher(0, self)
        self.schedule.add(self.teacher)
        
        # 创建初始学生
        self._next_student_id = 1
        for _ in range(n_initial):
            self._spawn_student()
        
        # 数据收集器
        self.datacollector = mesa.DataCollector(
            model_reporters={
                "Level": lambda m: m.teacher.L,
                "Permission": lambda m: m.teacher.U,
                "New_Ratio": "new_ratio",
                "Avg_Mastery_Old": "avg_mastery_old",
                "N_Total": "n_total",
                "N_Old": "n_old",
                "N_New": "n_new"
            },
            agent_reporters={
                "B": lambda a: getattr(a, "B", None),
                "E": lambda a: getattr(a, "E", None),
                "C": lambda a: getattr(a, "C", None),
                "Y": lambda a: getattr(a, "Y", None),
                "M_Current": lambda a: (getattr(a, "M", {}).get(
                    getattr(a.model.teacher, "L", None), None) 
                    if hasattr(a, "M") else None)
            }
        )
        
        # 运行时指标
        self.new_ratio = 0.0
        self.avg_mastery_old = 0.0
        self.n_total = 0
        self.n_old = 0
        self.n_new = 0
        self.step_count = 0
    
    def _spawn_student(self):
        """生成一个新学生"""
        B0 = self.random.uniform(10, 60)
        E0 = self.random.uniform(0.1, 0.5)
        s = StudentAgent(self._next_student_id, self, B0, E0)
        self.schedule.add(s)
        self._next_student_id += 1
    
    def add_new_students(self):
        """每节课按概率流入新学生"""
        if self.random.random() < self.new_student_rate:
            self._spawn_student()
    
    def step(self):
        """严格按照五阶段流程执行一节课"""
        
        # --- 阶段0：新学生入场（课前） ---
        self.add_new_students()
        
        # --- 阶段1：学生出勤 ---
        students = [a for a in self.schedule.agents if isinstance(a, StudentAgent)]
        for s in students:
            s.attend()
        
        N_t = [s for s in students if s.Y == 1]
        O_t = [s for s in N_t if not s.is_new()]
        
        self.n_total = len(N_t)
        self.n_old = len(O_t)
        self.n_new = self.n_total - self.n_old
        self.new_ratio = (self.n_new / self.n_total) if self.n_total > 0 else 1.0
        
        # --- 阶段2：老师临时决定本节课等级 ---
        self.teacher.decide_level(self.new_ratio)
        current_L = self.teacher.L
        
        # --- 阶段3：教学 ---
        for s in N_t:
            s.learn(current_L)
        
        # --- 阶段4：老师评估，决定下节课许可 ---
        self.avg_mastery_old = self.teacher.evaluate_and_permit(O_t, current_L)
        
        # --- 阶段5：学生属性演化 ---
        for s in students:
            s.update_attributes()
        
        # 收集数据
        self.datacollector.collect(self)
        self.step_count += 1
