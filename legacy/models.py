import numpy as np
from scipy.integrate import solve_ivp, ode
import sympy as sp
from collections import defaultdict


# Define the model dynamic equations
class EnergyModel:
    # Class level constants
    const_lambda = 863
    const_s = 0.024
    const_P_I_O2 = (760 - 47) * 0.21
    const_P_I_CO2 = (760 - 47) * 0.0006
    const_V_A = 3.28
    const_V_T = 38.9
    V_dot_0 = 4.28 / 60  # convert to seconds
    MP_O2 = 0.242 / 60  # convert to seconds
    MP_CO2 = 0.20 / 60  # convert to seconds
    const_GN = 5 / 60
    const_K2 = 0.2
    const_K3 = 0.046
    const_K4 = 0.016
    const_KT = 30  # KT = 30 s*L/s when I use cardiac output in L/s; in case of cardiac output in L/min, then it is 0.5 L
    const_mu = 0.35
    const_tau = 10
    const_RQ = 0.8

    # Activity_lists
    LWBM_list = ['Sitting', 'Standing', 'Reading', 'Survey'] # Low Whole-Body Motion
    HWBM_list = ['Stacking shelves', 'Dishwasher', 'Mopping'] # High Whole-Body Motion
    Walking_list = ['Stairs', 'Treadmill 3km', 'Treadmill 5km'] # Walking
    Cycling_list = ['Cycling'] # Cycling

    def __init__(self, parameters, data):
        # Initialize model with subject-specific parameters, trainable parameters, and data
        self.mass = parameters['mass']
        self.percentage_muscle = parameters['percentage_muscle']
        self.sex = parameters['sex']
        self.height = parameters['height']
        self.const_GPO2 = parameters['const_GPO2']
        self.const_GPCO2 = parameters['const_GPCO2']
        self.const_mu = parameters['const_mu']
        self.approach_nr = parameters['approach_nr']
        self.data = data

        # Load data
        self.t = data.loc[:, "t"]
        self.hr = data.loc[:, "hr"]
        self.vo2 = data.loc[:, "vo2"]
        self.vco2 = data.loc[:, "vco2"]
        self.velocity = data.loc[:, "velocity_DOT7"]
        self.acceleration = data.loc[:, "acceleration_DOT7"]
        self.velocity_DOT7 = data.loc[:, "velocity_DOT7"]
        self.velocity_DOT8 = data.loc[:, "velocity_DOT8"]
        self.velocity_DOT9 = data.loc[:, "velocity_DOT9"]

        self.gyration_DOT7 = data.loc[:, "gyration_DOT7"]
        self.gyration_DOT8 = data.loc[:, "gyration_DOT8"]
        self.gyration_DOT9 = data.loc[:, "gyration_DOT9"]
        # left and right wrists
        self.velocity_DOT5 = data.loc[:, 'velocity_DOT5']
        self.velocity_DOT6 = data.loc[:, 'velocity_DOT6']
        # left foot
        self.velocity_DOT10 = data.loc[:, 'velocity_DOT10']

        # activity labels
        self.activity_labels = data.loc[:, 'markers']

        # RQ reading mode
        self.RQ_mode = 'constant' # constant or adaptive
        # heart rate reading mode
        self.HR_mode = 'dynamic'
        self.HR_rest = 70
        self.HR_exercise = 130
        # cardiac output Q reading mode
        self.Q_mode = 'dynamic'
        if self.sex == 'male':
            self.Q_rest = 70 * 0.08975 / 60
            self.Q_exercise = 130 * 0.08975 / 60
        else:
            self.Q_rest = 70 * 0.06932 / 60
            self.Q_exercise = 130 * 0.06932 / 60
        # mu reading mode
        self.mu_mode = 'constant' # constant or adaptive
        self.factor_cycling = parameters['mu_correction_factors'][0]
        self.factor_lwbm = parameters['mu_correction_factors'][1]
        self.factor_hwbm = parameters['mu_correction_factors'][2]
        self.factor_walking = parameters['mu_correction_factors'][3]

        # Initialize lists to store intermediate values
        self.muscle_mu_values = []
        self.P_a_O2_values = []
        self.P_a_CO2_values = []
        self.V_dot_A_values = []
        self.C_a_O2_values = []  # arterial concentration in liters O2 / liter of blood
        self.C_a_CO2_values = []  # arterial concentration in liters CO2 / liter of blood
        self.C_v_O2_values = []  # venous concentration in liter O2 / per liter blood
        self.C_v_CO2_values = []  # venous concentration in liter CO2 / per liter blood
        self.C_e_O2_values = []  # end-capillary concentration in liter O2 / liter of blood
        self.C_e_CO2_values = []  # end-capillary concentration in liter CO2 / liter of blood
        self.t_values = []
        self.VO2_values = []
        self.VCO2_values = []
        self.V_dot_D_values = []
        self.Q_values = []
        self.energy_values = []
        self.MPO2_values = []  # metabolic production rate of oxygen (R_{M, O2} = E(t) / 19.6)
        self.MPCO2_values = [] # metabolic production rate of carbon dioxide (R_{M,CO2})

        # Subject-dependent variables
        self.SV = 0.08975 if self.sex == 'male' else 0.06932  # stroke volumn
        self.const_V_T = self.mass * self.percentage_muscle / 100 / 1.06  # 1.06 is the muscle tissue density, in kg/L

        # Calculate const_K1
        self.const_K1 = -(-self.const_GPO2 * 0.19603592627610356 + self.const_GPCO2 * 40.0)

    # Dissociation curves
    def dissociation_curve_O2(self, P_a_O2):
        P_a_O2 = np.clip(P_a_O2, 30.0, 200.0)
        C = self.const_K2 * ((1 - np.exp(-self.const_K3 * P_a_O2)) ** 2)
        return C

    def dissociation_curve_CO2(self, P_a_CO2):
        C = self.const_K4 * P_a_CO2
        return C

    @staticmethod
    def weir_equation(vo2, vco2):
        return 3.9 * vo2 + 1.1 * vco2

class ODEModel(EnergyModel):
    # Remco's methods
    def store_intermediate(self, t):
        # Calculate VO2, VCO2 and V_dot_D
        VO2, VCO2 = self.calc_ventilation()

        # Store intermediate values in separate lists
        self.P_a_O2_values = np.append(self.P_a_O2_values, self.P_a_O2)
        self.P_a_CO2_values = np.append(self.P_a_CO2_values, self.P_a_CO2)
        self.V_dot_A_values = np.append(self.V_dot_A_values, self.V_dot_A)
        self.C_a_O2_values = np.append(self.C_a_O2_values, self.C_a_O2)
        self.C_a_CO2_values = np.append(self.C_a_CO2_values, self.C_a_CO2)
        self.C_v_O2_values = np.append(self.C_v_O2_values, self.C_v_O2)
        self.C_v_CO2_values = np.append(self.C_v_CO2_values, self.C_v_CO2)
        self.C_e_O2_values = np.append(self.C_e_O2_values, self.C_e_O2)
        self.C_e_CO2_values = np.append(self.C_e_CO2_values, self.C_e_CO2)
        self.VO2_values = np.append(self.VO2_values, VO2)
        self.VCO2_values = np.append(self.VCO2_values, VCO2)
        self.MPO2_values = np.append(self.MPO2_values, self.MP_O2)
        self.MPCO2_values = np.append(self.MPCO2_values, self.MP_CO2)
        self.t_values = np.append(self.t_values, t)
        self.Q_values = np.append(self.Q_values, self.Q)
        self.energy_values = np.append(self.energy_values, self.energy)

    def update_parameters_perstep(self, t):
        t = int(t)
        hr = self.hr.iloc[t]
        SV = self.SV

        # Change Q
        self.Q = hr * SV / 60  # Convert to seconds

        # Change MP_O2 and MP_CO2
        MP_O2, MP_CO2 = self.velocity_to_MP_perstep(t)
        self.MP_O2 = MP_O2
        self.MP_CO2 = MP_CO2

    def velocity_to_MP_perstep(self, t):

        approach_nr = self.approach_nr
        # Only pelvis
        if approach_nr == 1:
            # kinetic_energy = abs(0.5 * self.mass * (self.velocity.iloc[t] ** 2))  # Remco's original implementation
            kinetic_energy = 0.5 * self.mass * (self.velocity.iloc[t]**2)
            # kinetic_energy_1 = 0.5*self.mass*(self.velocity.iloc[t]**2)
            # kinetic_energy_2 = self.mass * np.sum(self.velocity.iloc[:t+1]*self.acceleration.iloc[:t+1]) * 1
            # kinetic_energy = kinetic_energy_1 + kinetic_energy_2

        # Pelvis, both legs
        elif approach_nr == 2:
            mass = self.mass
            mass_torso = mass * 0.6664 if self.sex == 'male' else mass * 0.6314
            mass_leg = mass * 0.1668 if self.sex == 'male' else mass * 0.1843
            KE_torso = 0.5 * mass_torso * (self.velocity_DOT7.iloc[t] ** 2)
            KE_right_leg = 0.5 * mass_leg * (self.velocity_DOT8.iloc[t] ** 2)
            KE_left_leg = 0.5 * mass_leg * (self.velocity_DOT9.iloc[t] ** 2)

            kinetic_energy = KE_torso + KE_left_leg + KE_right_leg

        # Pelvis, both legs, with gyrations included
        elif approach_nr == 3:
            mass = self.mass
            height = self.height / 100  # meters
            mass_torso = mass * 0.8332 if self.sex == 'male' else mass * 0.8157
            mass_leg = mass * 0.1668 if self.sex == 'male' else mass * 0.1843
            # Segment length
            length_torso = 0.3 * height
            length_leg = 0.45 * height
            # Moments of inertia
            I_torso = (1 / 12) * mass_torso * (length_torso ** 2)
            I_leg = (1 / 3) * mass_leg * (length_leg ** 2)
            # Translational KE
            KE_torso = 0.5 * mass_torso * (self.velocity_DOT7.iloc[t] ** 2)
            KE_right_leg = 0.5 * mass_leg * (self.velocity_DOT8.iloc[t] ** 2)
            KE_left_leg = 0.5 * mass_leg * (self.velocity_DOT9.iloc[t] ** 2)
            # Rotational KE
            KE_rot_torso = 0.5 * I_torso * (self.gyration_DOT7.iloc[t] ** 2)
            KE_rot_right_leg = 0.5 * I_leg * (self.gyration_DOT8.iloc[t] ** 2)
            KE_rot_left_leg = 0.5 * I_leg * (self.gyration_DOT9.iloc[t] ** 2)

            kinetic_energy = (KE_torso + KE_right_leg + KE_left_leg +
                              KE_rot_torso + KE_rot_right_leg + KE_rot_left_leg)

        energy = kinetic_energy / 1000

        # Account for efficiency
        if self.mu_mode == 'constant':
            self.energy = energy / self.const_mu
        elif self.mu_mode == 'adaptive':
            activity_label = self.activity_labels.iloc[t]
            if activity_label in self.Cycling_list:
                adapted_mu = self.const_mu / self.factor_cycling
            elif activity_label in self.HWBM_list:
                adapted_mu = self.const_mu / self.factor_hwbm
            elif activity_label in self.LWBM_list:
                adapted_mu = self.const_mu / self.factor_lwbm
            elif activity_label in self.Walking_list:
                adapted_mu = self.const_mu / self.factor_walking
            else:
                adapted_mu = self.const_mu
            self.energy = energy / adapted_mu

        if self.energy < 0:
            self.energy = 0

        MP_O2 = self.energy / 19.6
        MP_CO2 = self.const_RQ * MP_O2

        return MP_O2, MP_CO2

    def model(self, t, y):
        P_a_O2, P_a_CO2, C_v_O2, C_v_CO2, V_dot_A = y

        C_e_O2 = self.dissociation_curve_O2(P_a_O2)
        C_e_CO2 = self.dissociation_curve_CO2(P_a_CO2)
        V_dot_A = V_dot_A if V_dot_A > 0 else 0

        # Mass balance in lungs
        # This is where the heart rate data come into play
        # Q = HR * SV
        dpa_O2_dt = (V_dot_A * (self.const_P_I_O2 - P_a_O2) + self.const_lambda * self.Q * (1 - self.const_s) * (
                    C_v_O2 - C_e_O2)) / self.const_V_A
        dpa_CO2_dt = (V_dot_A * (self.const_P_I_CO2 - P_a_CO2) + self.const_lambda * self.Q * (1 - self.const_s) * (
                    C_v_CO2 - C_e_CO2)) / self.const_V_A

        C_a_O2 = (1 - self.const_s) * C_e_O2 + self.const_s * C_v_O2
        C_a_CO2 = (1 - self.const_s) * C_e_CO2 + self.const_s * C_v_CO2

        # Respiratory controller
        dvdot_p_dt = (-self.const_GPO2 * C_a_O2 + self.const_GPCO2 * P_a_CO2 + self.const_K1)

        # Respiratory controller updated
        # T = self.const_KT / self.Q
        # T = int(np.round(T))
        # if len(self.t_values) > 0:
        #     if T > self.t_values[-1] + 1:
        #         C_a_O2_delayed  = C_a_O2
        #         P_a_CO2_delayed = P_a_CO2
        #     else:
        #         C_a_O2_delayed = self.C_a_O2_values[-1 - T + 2]
        #         P_a_CO2_delayed = self.P_a_CO2_values[-1 - T + 2]
        # else:
        #     C_a_O2_delayed = C_a_O2
        #     P_a_CO2_delayed = P_a_CO2
        #
        # dvdot_p_dt = (-self.const_GPO2 * C_a_O2_delayed +
        #               self.const_GPCO2 * P_a_CO2_delayed + self.const_K1 - V_dot_A) / self.const_tau

        # Mass balance in tissue
        # This is where the IMU measurements come into play
        # MP_O2 = R_{M,O2}
        dcv_O2_dt = (self.Q * (C_a_O2 - C_v_O2) - self.MP_O2) / self.const_V_T
        dcv_CO2_dt = (self.Q * (C_a_CO2 - C_v_CO2) + self.MP_CO2) / self.const_V_T

        # Remco's implementation:
        if C_v_O2 + dcv_O2_dt < 0:
            dcv_O2_dt = -C_v_O2
        if C_v_CO2 + dcv_CO2_dt < 0:
            dcv_CO2_dt = -C_v_CO2

        # Store values
        self.P_a_O2 = P_a_O2
        self.P_a_CO2 = P_a_CO2
        self.V_dot_A = V_dot_A
        self.C_a_O2 = C_a_O2
        self.C_a_CO2 = C_a_CO2
        self.C_v_O2 = C_v_O2
        self.C_v_CO2 = C_v_CO2
        self.C_e_O2 = C_e_O2
        self.C_e_CO2 = C_e_CO2

        return np.array([dpa_O2_dt, dpa_CO2_dt, dcv_O2_dt, dcv_CO2_dt, dvdot_p_dt])

    def integrate_step(self, solver, max_step):
        # Update the parameters based on the current time
        self.update_parameters_perstep(solver.t)

        # Perform one step of integration
        solver.integrate(solver.t + max_step)  # Perform one step with the specified max step size

        # Store intermediate values
        self.store_intermediate(solver.t)

    def integrate_model(self, t_initial, t_final, max_step):
        # Define the initial conditions
        # P_a_O2, P_a_CO2, C_v_O2, C_v_CO2, V_dot_A = y
        initial_conditions = np.array([100.2, 40, 0.144, 0.610, (4.28 / 60)])

        # Initialize the ODE solver
        solver = ode(self.model)
        solver.set_integrator('dopri5', max_step=max_step, nsteps=10000)
        solver.set_initial_value(initial_conditions, t_initial)

        # Integrate the model over the time span
        while solver.successful() and solver.t < t_final:
            self.integrate_step(solver, max_step)

    def calc_ventilation(self):
        V_dot_A = self.V_dot_A

        VO2 = (V_dot_A * (0.21 - (self.P_a_O2 / (760 - 47))))
        if (self.P_a_CO2 / (760 - 47)) > 0.0004:
            VCO2 = -1 * (V_dot_A * (0.0004 - (self.P_a_CO2 / (760 - 47))))
        else:
            VCO2 = 0

        return VO2, VCO2

class EKFModel(EnergyModel):
    def __init__(self, parameters, data):
        super().__init__(parameters, data)
        self.controller_mode = 'delayed' # delayed or instant
        self.state_variables = [r"$P_{a,O_2}$", r"$P_{a,CO_2}$", r"$C_{v,O_2}$", r"$C_{v,CO_2}$", r"$\dot{V}_A$"]
        self.CO_values = [] # estimated cardiac output values from VO2
        self.raw_SV_values = []
        self.smoothed_SV_values = []
        self.raw_VO2_values = []
        self.smoothed_VO2_values = []

    def store_intermediate(self, t):
        # Store intermediate values in separate lists
        self.P_a_O2_values = np.append(self.P_a_O2_values, self.P_a_O2)
        self.P_a_CO2_values = np.append(self.P_a_CO2_values, self.P_a_CO2)
        self.V_dot_A_values = np.append(self.V_dot_A_values, self.V_dot_A)
        self.C_a_O2_values = np.append(self.C_a_O2_values, self.C_a_O2)
        self.C_a_CO2_values = np.append(self.C_a_CO2_values, self.C_a_CO2)
        self.C_v_O2_values = np.append(self.C_v_O2_values, self.C_v_O2)
        self.C_v_CO2_values = np.append(self.C_v_CO2_values, self.C_v_CO2)
        self.C_e_O2_values = np.append(self.C_e_O2_values, self.C_e_O2)
        self.C_e_CO2_values = np.append(self.C_e_CO2_values, self.C_e_CO2)
        self.MPO2_values = np.append(self.MPO2_values, self.MP_O2)
        self.MPCO2_values = np.append(self.MPCO2_values, self.MP_CO2)
        self.t_values = np.append(self.t_values, t)

    # Derivative of dissociation curve: dC/dP = 2 * K2 * K3 * exp(-K3 * P) * (1 - exp(-K3 * P))
    def d_dissociation_curve_O2(self, P_a_O2):
        P_a_O2 = np.clip(P_a_O2, 30.0, 200.0) # clip values within physiological reason for numerical stability
        exp_term = np.exp(-self.const_K3 * P_a_O2)
        return 2 * self.const_K2 * self.const_K3 * exp_term * (1 - exp_term)

    def d_dissociation_curve_CO2(self, P_a_CO2):
        return self.const_K4

    def calc_ventilation_ekf(self, V_dot_A, P_a_O2, P_a_CO2):
        VO2 = V_dot_A * (0.21 - (P_a_O2 / (760 - 47)))
        VCO2 = -1 * (V_dot_A * (0.0006 - (P_a_CO2 / (760 - 47))))  # should be a positive value

        # convert from BTPS -> STPD (used for Weir's equation / metabolic rate)
        factor_B = (760 - 47) / 760 * 273 / 310
        VO2 = factor_B * VO2
        VCO2 = factor_B * VCO2

        VO2 = max(VO2, 0)
        VCO2 = max(VCO2, 0)

        return VO2, VCO2

    # Extended Kalman Filter Implementations
    # f(x, Q) dynamics
    def ode_rhs(self, x, Q_t):
        x1, x2, x3, x4, x5 = x  # unpack state: P_a_O2, P_a_CO2, C_v_O2, C_v_CO2, V_dot_A

        # convert from BTPS -> STPD (used for Weir's equation / metabolic rate)
        # factor_B = (760 - 47) / 760 * 273 / 310

        Ce_O2 = self.dissociation_curve_O2(x1)
        Ce_CO2 = self.dissociation_curve_CO2(x2)

        Ca_O2 = (1 - self.const_s) * Ce_O2 + self.const_s * x3
        Ca_CO2 = (1 - self.const_s) * Ce_CO2 + self.const_s * x4

        dx1 = (x5 * (self.const_P_I_O2 - x1) + self.const_lambda * Q_t * (1 - self.const_s) * (x3 - Ce_O2)) / self.const_V_A
        dx2 = (x5 * (self.const_P_I_CO2 - x2) + self.const_lambda * Q_t * (1 - self.const_s) * (x4 - Ce_CO2)) / self.const_V_A
        VO2, VCO2 = self.calc_ventilation_ekf(x5, x1, x2)
        dx3 = (Q_t * (Ca_O2 - x3)) / self.const_V_T - VO2
        dx4 = (Q_t * (Ca_CO2 - x4)) / self.const_V_T + VCO2

        if self.controller_mode == 'delayed':
            # Respiratory controller updated (with time delay)
            T = self.const_KT / Q_t
            T = int(np.round(T))
            if len(self.t_values) > 0:
                if T > self.t_values[-1] + 1:
                    C_a_O2_delayed = Ca_O2
                    P_a_CO2_delayed = x2
                else:
                    C_a_O2_delayed = self.C_a_O2_values[-1 - T + 2]
                    P_a_CO2_delayed = self.P_a_CO2_values[-1 - T + 2]
            else:
                C_a_O2_delayed = Ca_O2
                P_a_CO2_delayed = x2

            dx5 = (-self.const_GPO2 * C_a_O2_delayed +
                   self.const_GPCO2 * P_a_CO2_delayed + self.const_K1 - x5) / self.const_tau
        else:
            # original respiratory controller
            dx5 = -self.const_GPO2 * Ca_O2 + self.const_GPCO2 * x2 + self.const_K1

        # Remco's implementation:
        # Physically: Concentrations can’t be negative — this is a hard physical constraint.
        # Numerically: Prevents the solver from diverging or producing biologically invalid states.
        # Mathematically: You're not altering the dynamics arbitrarily, only truncating them to prevent unphysical values.
        # Stability: Helps avoid overflow or instability in equations like the oxygen dissociation curve.
        if x3 + dx3 < 0:
            dx3 = -x3
        if x4 + dx4 < 0:
            dx4 = -x4

        # Store values
        self.P_a_O2 = x1
        self.P_a_CO2 = x2
        self.V_dot_A = x5
        self.C_a_O2 = Ca_O2
        self.C_a_CO2 = Ca_CO2
        self.C_v_O2 = x3
        self.C_v_CO2 = x4
        self.C_e_O2 = Ce_O2
        self.C_e_CO2 = Ce_CO2
        self.C_a_O2_delayed = C_a_O2_delayed
        self.P_a_CO2_delayed = P_a_CO2_delayed

        return np.array([dx1, dx2, dx3, dx4, dx5])

    def compute_f_jacobian_chainrule_items(self, x, hr_t, vo2_t):
        x1, x2, x3, x4, x5 = x

        Ce_O2 = self.dissociation_curve_O2(x1)
        Ce_CO2 = self.dissociation_curve_CO2(x2)

        Ca_O2 = (1 - self.const_s) * Ce_O2 + self.const_s * x3
        Ca_CO2 = (1 - self.const_s) * Ce_CO2 + self.const_s * x4

        dfdu = np.zeros((5, 1))
        dudx = np.zeros((1, 5))

        dfdu[0, 0] = self.const_lambda * (1 - self.const_s) / self.const_V_A * (x3 - Ce_O2)
        dfdu[1, 0] = self.const_lambda * (1 - self.const_s) / self.const_V_A * (x4 - Ce_CO2)
        dfdu[2, 0] = (Ca_O2 - x3) / self.const_V_T
        dfdu[3, 0] = (Ca_CO2 - x4) / self.const_V_T

        dudx[0, 0] = -hr_t / 60 * 0.02 * x5 / ((760 - 47) * vo2_t)
        dudx[0, 4] = hr_t / 60 * 0.02 * (0.21 - x1 / (760 - 47)) / vo2_t

        return dfdu @ dudx

    def compute_f_jacobian(self, x, Q_t):
        x1, x2, x3, x4, x5 = x

        factor_A = 760 - 47
        factor_B = factor_A / 760 * 273 / 310

        dCe_O2 = self.d_dissociation_curve_O2(x1)
        # dCe_CO2 = self.const_K4
        dCe_CO2 = self.d_dissociation_curve_CO2(x2)

        F = np.zeros((5, 5))
        # ∂f1/∂x1
        F[0, 0] = (-x5 - self.const_lambda * Q_t * (1 - self.const_s) * dCe_O2) / self.const_V_A
        # ∂f1/∂x3
        F[0, 2] = self.const_lambda * Q_t * (1 - self.const_s) / self.const_V_A
        # ∂f1/∂x5
        F[0, 4] = (self.const_P_I_O2 - x1) / self.const_V_A

        # ∂f2/∂x2
        # updated
        # F[1, 1] = (-x5 - self.const_lambda * Q_t * (1 - self.const_s) * dCe_O2) / self.const_V_A
        F[1, 1] = (-x5 - self.const_lambda * Q_t * (1 - self.const_s) * dCe_CO2) / self.const_V_A
        # ∂f2/∂x4
        F[1, 3] = self.const_lambda * Q_t * (1 - self.const_s) / self.const_V_A
        # ∂f2/∂x5
        F[1, 4] = (self.const_P_I_CO2 - x2) / self.const_V_A

        # ∂f3/∂x1
        # updated
        # F[2, 0] = Q_t * (1 - self.const_s) * dCe_O2 / self.const_V_T
        F[2, 0] = (Q_t * (1 - self.const_s) * dCe_O2 + factor_B * x5 / factor_A) / self.const_V_T
        # ∂f3/∂x3
        F[2, 2] = (Q_t / self.const_V_T) * (self.const_s - 1)
        # ∂f3/∂x5
        # added
        F[2, 4] = (- factor_B * 0.21 + factor_B * x1 / factor_A) / self.const_V_T

        # ∂f4/∂x2
        # updated
        # F[3, 1] = Q_t * (1 - self.const_s) * dCe_CO2 / self.const_V_T
        F[3, 1] = (Q_t * (1 - self.const_s) * dCe_CO2 + factor_B * x5 / factor_A) / self.const_V_T
        # ∂f4/∂x4
        F[3, 3] = (Q_t / self.const_V_T) * (self.const_s - 1)
        # ∂f4/∂x5
        # added
        F[3, 4] = (- factor_B * 0.0006 + factor_B * x2 / factor_A) / self.const_V_T

        if self.controller_mode == 'delayed':
            # With time delay: treat C_aO2(t-T), x2(t-T) as exogenous (no dependence on current x)
            # ∂f5/∂x5
            F[4, 4] = - 1 / self.const_tau
        else:
            # For respiratory controller without time delay
            # No time delay: controller depends on current CaO2(x1,x3) and x2
            F[4, 0] = -(self.const_GPO2 / self.const_tau) * (1 - self.const_s) * dCe_O2  # ∂f5/∂x1
            F[4, 1] = (self.const_GPCO2 / self.const_tau)  # ∂f5/∂x2
            F[4, 2] = -(self.const_GPO2 / self.const_tau) * self.const_s  # ∂f5/∂x3
            F[4, 4] = -1.0 / self.const_tau  # ∂f5/∂x5

        return F

    def compute_F_state_transition_matrix(self, x, Q_t, hr_t, vo2_t, delta_t):
        # continuous-time dynamic Jacobian
        f_jacobian = self.compute_f_jacobian(x, Q_t) + self.compute_f_jacobian_chainrule_items(x, hr_t, vo2_t)
        # state-transition matrix for discrete-time
        F_jacobian = f_jacobian * delta_t + np.eye(5)

        return F_jacobian

    def measurement_model(self, x, Q_t):
        x1, x2, x3, x4, _ = x
        Ce_O2 = self.dissociation_curve_O2(x1)
        Ce_CO2 = self.dissociation_curve_CO2(x2)

        base_O2 = Q_t * (1 - self.const_s) * (Ce_O2 - x3)
        base_CO2 = Q_t * (1 - self.const_s) * (Ce_CO2 - x4)
        return np.array([base_O2, base_CO2])
        # return base_O2

    def compute_h_jacobian_chainrule_items(self, x, hr_t, vo2_t):
        x1, x2, x3, x4, x5 = x

        Ce_O2 = self.dissociation_curve_O2(x1)
        Ce_CO2 = self.dissociation_curve_CO2(x2)

        dhdu = np.zeros((2, 1))
        dudx = np.zeros((1, 5))

        dhdu[0, 0] = (1 - self.const_s) * (Ce_O2 - x3)
        dhdu[1, 0] = (1 - self.const_s) * (Ce_CO2 - x4)

        dudx[0, 0] = -hr_t / 60 * 0.02 * x5 / ((760 - 47) * vo2_t)
        dudx[0, 4] = hr_t / 60 * 0.02 * (0.21 - x1 / (760 - 47)) / vo2_t

        return dhdu @ dudx

    def compute_H_jacobian(self, x, Q_t, hr_t, vo2_t):
        x1, x2, _, _, _ = x

        dCeO2_dx1 = self.d_dissociation_curve_O2(x1)
        dCeCO2_dx2 = self.d_dissociation_curve_CO2(x2)
        H = np.zeros((2, 5))
        factor = Q_t * (1 - self.const_s)
        # ∂z1/∂x1 and ∂z1/∂x3
        H[0, 0] = factor * dCeO2_dx1
        H[0, 2] = -factor

        # ∂z2/∂x2 and ∂z2/∂x4
        H[1, 1] = factor * dCeCO2_dx2
        H[1, 3] = -factor

        H = H + self.compute_h_jacobian_chainrule_items(x, hr_t, vo2_t)

        return H

    def ekf_update(self, x_pred, P_pred, z_meas, Q_t, hr_t, vo2_t, R_k):
        H = self.compute_H_jacobian(x_pred, Q_t, hr_t, vo2_t) # x_{t|t-1}, Q_{t} or Q_{t|t-1}

        z_pred = self.measurement_model(x_pred, Q_t) # x_{t|t-1}
        y = z_meas - z_pred
        # y = np.array([[z_meas - z_pred]])
        S = H @ P_pred @ H.T + R_k
        K = P_pred @ H.T @ np.linalg.inv(S)

        x_upd = x_pred + K @ y
        # x_upd = x_pred + (K @ y).flatten()

        # P_upd = (np.eye(len(x_pred)) - K @ H) @ P_pre

        # Joseph's form, numerically safer
        I = np.eye(len(x_pred))
        P_upd = (I - K @ H) @ P_pred @ (I - K @ H).T + K @ R_k @ K.T
        # optional: enforce symmetry against roundoff
        P_upd = 0.5 * (P_upd + P_upd.T)

        return x_upd, P_upd

    def update_parameters(self):
        hr = self.hr
        SV = self.SV

        # Change Q
        if self.Q_mode == 'dynamic':
            self.Q = hr * SV / 60  # Convert to seconds
        else:
            self.Q = np.zeros(len(hr))
            for t_index in range(len(self.activity_labels)):
                if self.activity_labels[t_index] in self.Cycling_list:
                    self.Q[t_index] = self.Q_exercise
                elif self.activity_labels[t_index] in self.LWBM_list:
                    self.Q[t_index] = self.Q_rest
                elif self.activity_labels[t_index] in self.HWBM_list:
                    self.Q[t_index] = self.Q_exercise
                elif self.activity_labels[t_index] in self.Walking_list:
                    self.Q[t_index] = self.Q_exercise
                else:
                    self.Q[t_index] = self.Q_rest

        # Change HR
        if self.HR_mode == 'static':
            self.hr = np.zeros(len(hr)) + self.HR_rest
            # for t_index in range(len(self.activity_labels)):
            #     if self.activity_labels[t_index] in self.Cycling_list:
            #         self.hr[t_index] = self.HR_exercise
            #     elif self.activity_labels[t_index] in self.LWBM_list:
            #         self.hr[t_index] = self.HR_rest
            #     elif self.activity_labels[t_index] in self.HWBM_list:
            #         self.hr[t_index] = self.HR_exercise
            #     elif self.activity_labels[t_index] in self.Walking_list:
            #         self.hr[t_index] = self.HR_exercise
            #     else:
            #         self.hr[t_index] = self.HR_rest


        # Change MP_O2 and MP_CO2
        MP_O2, MP_CO2 = self.velocity_to_MP()
        self.MP_O2_values = MP_O2
        self.MP_CO2_values = MP_CO2

    def velocity_to_MP(self):

        approach_nr = self.approach_nr

        # Only pelvis
        if approach_nr == 1:
            kinetic_energy = 0.5 * self.mass * (self.velocity ** 2)

        # Pelvis, both legs
        # This approach tends to overestimate treadmill 3km and treadmill 5km, while underestimate cycling
        elif approach_nr == 2:
            mass = self.mass
            mass_torso = mass * 0.6664 if self.sex == 'male' else mass * 0.6314
            mass_leg = mass * 0.1668 if self.sex == 'male' else mass * 0.1843
            KE_torso = 0.5 * mass_torso * (self.velocity_DOT7 ** 2)
            KE_right_leg = 0.5 * mass_leg * (self.velocity_DOT8 ** 2)
            KE_left_leg = 0.5 * mass_leg * (self.velocity_DOT9 ** 2)

            kinetic_energy = KE_torso + KE_left_leg + KE_right_leg

        # Pelvis, both legs, with gyrations included
        elif approach_nr == 3:
            mass = self.mass
            height = self.height / 100 # meters
            mass_torso = mass * 0.8332 if self.sex == 'male' else mass * 0.8157
            mass_leg = mass * 0.1668 if self.sex == 'male' else mass * 0.1843
            # Segment length
            length_torso = 0.3 * height
            length_leg = 0.45 * height
            # Moments of inertia
            I_torso = (1 / 12) * mass_torso * (length_torso ** 2)
            I_leg = (1 / 3) * mass_leg * (length_leg ** 2)
            # Translational KE
            KE_torso = 0.5 * mass_torso * (self.velocity_DOT7 ** 2)
            KE_right_leg = 0.5 * mass_leg * (self.velocity_DOT8 ** 2)
            KE_left_leg = 0.5 * mass_leg * (self.velocity_DOT9 ** 2)
            # Rotational KE
            KE_rot_torso = 0.5 * I_torso * (self.gyration_DOT7 ** 2)
            KE_rot_right_leg = 0.5 * I_leg * (self.gyration_DOT8 ** 2)
            KE_rot_left_leg = 0.5 * I_leg * (self.gyration_DOT9 ** 2)

            kinetic_energy = ( KE_torso + KE_right_leg + KE_left_leg +
                               KE_rot_torso + KE_rot_right_leg + KE_rot_left_leg)

        elif self.approach_nr == 4:
            mass = self.mass
            mass_torso = mass * 0.5664 if self.sex == 'male' else mass * 0.5314
            mass_leg = mass * 0.2168 if self.sex == 'male' else mass * 0.2343
            # mass_wrist = mass * 0.022 / 2 if self.sex == 'male' else mass * 0.019 / 2
            # mass_foot = 0.0145 * mass if self.sex == 'male' else 0.0132 * mass
            KE_torso = 0.5 * mass_torso * (self.velocity_DOT7 ** 2)
            KE_right_leg = 0.5 * mass_leg * (self.velocity_DOT8 ** 2)
            KE_left_leg = 0.5 * mass_leg * (self.velocity_DOT9 ** 2)
            # KE_left_foot = 0.5 * mass_foot * (self.velocity_DOT10 ** 2)
            # KE_right_wrist = 0.5 * mass_wrist * (self.velocity_DOT6 ** 2)
            # KE_left_wrist = 0.5 * mass_wrist * (self.velocity_DOT5 ** 2)

            kinetic_energy = KE_torso + KE_right_leg + KE_left_leg

        energy = kinetic_energy / 1000

        # Account for efficiency

        # fixed mu approach
        if self.mu_mode == 'constant':
            self.energy = energy / self.const_mu
        elif self.mu_mode == 'adaptive':
            # adaptive mu values for muscle efficieny for different activities
            const_mu_list = np.array([self.const_mu[0]] * len(energy))
            for t_index in range(len(self.activity_labels)):
                if self.activity_labels[t_index] in self.Cycling_list:
                    const_mu_list[t_index] = self.const_mu[1] / self.factor_cycling
                elif self.activity_labels[t_index] in self.LWBM_list:
                    const_mu_list[t_index] = self.const_mu[2] / self.factor_lwbm
                elif self.activity_labels[t_index] in self.HWBM_list:
                    const_mu_list[t_index] = self.const_mu[3] / self.factor_hwbm
                elif self.activity_labels[t_index] in self.Walking_list:
                    const_mu_list[t_index] = self.const_mu[4] / self.factor_walking

            self.energy = energy / const_mu_list

        self.energy = np.array([0 if a < 0 else a for a in self.energy])

        MP_O2 = self.energy / 19.6

        if self.RQ_mode == 'constant':
            MP_CO2 = self.const_RQ * MP_O2
        elif self.RQ_mode == 'adaptive':
            RQ_list = np.array([0.7] * len(energy))
            for t_index in range(len(self.activity_labels)):
                if self.activity_labels[t_index] in self.Cycling_list:
                    RQ_list[t_index] = 0.8
                elif self.activity_labels[t_index] in self.LWBM_list:
                    RQ_list[t_index] = 0.7
                elif self.activity_labels[t_index] in self.HWBM_list:
                    RQ_list[t_index] = 0.8
                elif self.activity_labels[t_index] in self.Walking_list:
                    RQ_list[t_index] = 0.8
            MP_CO2 = RQ_list * MP_O2

        return MP_O2, MP_CO2

    def VO2_to_SV(self, vo2_value, lambda_1=0.02, lambda_2=0.08975):
        '''
        :param vo2_value: oxygen uptake (L/min)
        :param lambda_1: contractility (controls how fast SV rises with intensity), typical 0.01–0.03
        :param lambda_2: baseline stroke volume
        :return: cardiac output (CO)
        '''
        vo2_value = vo2_value * 60 # convert from L/s to L/min
        # Ensure positive VO2 (avoid log(0))
        vo2_value = np.maximum(vo2_value, 1e-6)
        SV_value = lambda_1 * np.log(vo2_value) + lambda_2

        return SV_value

    # === EKF main loop ===
    def run_ekf(self, x0, P0, u_series, MP_O2CO2_series, t_series, N_k, R_k):
        x_estimates = [x0]
        P_estimates = [P0]
        x_i = x0
        P_i = P0
        vo2co2_est_list = []
        vo2_est_smoothed_list = []
        SV_est_smoothed_list = []

        MP_O2_series, MP_CO2_series = MP_O2CO2_series

        # initialise values
        VO2_estimate, VCO2_estimate = self.calc_ventilation_ekf(x_i[-1], x_i[0], x_i[1])
        vo2co2_est_list.append([VO2_estimate, VCO2_estimate])
        vo2_est_smoothed_list.append(VO2_estimate)
        self.smoothed_VO2_values.append(VO2_estimate)

        sv_raw = self.VO2_to_SV(VO2_estimate)
        SV_est_smoothed_list.append(sv_raw)

        for k in range(1, len(t_series)):
            # integrated value at time step k, starting from 1
            # integration starts from time step 0
            t_span = (t_series[k - 1], t_series[k])
            delta_t = t_span[1] - t_span[0]
            i = k - 1  # inputs/index at start of interval

            hr_i = u_series[i]
            # raw VO2 (L/s): use previous estimate or a prior at k=0
            vo2_raw = vo2co2_est_list[i][0]
            self.raw_VO2_values.append(vo2_raw)
            # 1) causal EMA on VO2 (no window needed)
            tau = 10.0  # seconds (try 5–15)
            alpha = np.exp(-delta_t/ tau)  # dt = 1 s
            vo2_eps = 0.0008  # vo2 floor value
            if i > 0:
                vo2_i = alpha * vo2_est_smoothed_list[i - 1] + (1 - alpha) * max(vo2_raw, vo2_eps)
            else:
                vo2_i = max(vo2_raw, vo2_eps)
            vo2_est_smoothed_list.append(vo2_i)
            self.smoothed_VO2_values.append(vo2_i)
            # 2) SV from smoothed VO2 (convert to L/min for the SV law)
            # we use VO2 estimated from previous time step to calculate the stroke volume
            # in this case, the stroke volume can be treated as a constant (thanks to the delayed term) and serves
            # as the external input to our state-space model.
            sv_raw = self.VO2_to_SV(vo2_i)  # in L/beat
            self.raw_SV_values.append(sv_raw)
            # 3) inertia + physiological bounds on SV
            beta = 0.1  # 0.05–0.2 works well
            if i > 0:
                SV_i = SV_est_smoothed_list[i - 1] + beta * (sv_raw - SV_est_smoothed_list[i - 1])
            else:
                SV_i = sv_raw
            SV_i = np.clip(SV_i, 0.06, 0.15)  # L/beat
            SV_est_smoothed_list.append(SV_i)
            self.smoothed_SV_values.append(SV_i)
            # 4) form Q (keep delay consistent; here both HR & SV are delayed)
            Q_i = hr_i * SV_i / 60.0  # L/s
            self.CO_values.append(Q_i)

            # Euler's Method
            # predict from i → i+1
            x_pred = x_i + delta_t * self.ode_rhs(x_i, Q_i)
            x_pred[4] = max(x_pred[4], 0.0)  # ensure V_dot_A is non-negative

            F = self.compute_F_state_transition_matrix(x_i, Q_i, hr_i, vo2_i, delta_t)  # state-transition matrix at x_{t-1|t-1}
            P_pred = F @ P_i @ F.T + N_k

            # It is important that the MP_CO2_series should be negative/flipped values due to the measurement function's definition
            z_k = np.array([MP_O2_series[i+1], -MP_CO2_series[i+1]])
            # derive Q at time step t|t-1 based on predicted state X_{t|t-1}
            vo2_pred, _ = self.calc_ventilation_ekf(x_pred[-1], x_pred[0], x_pred[1])
            vo2_pred = alpha * vo2_est_smoothed_list[i] + (1 - alpha) * max(vo2_pred, vo2_eps)
            SV_pred = self.VO2_to_SV(vo2_pred)  # in L/beat
            SV_pred = SV_est_smoothed_list[i] + beta * (SV_pred - SV_est_smoothed_list[i])
            SV_pred = np.clip(SV_pred, 0.06, 0.15)  # L/beat
            hr_pred = u_series[i+1]
            Q_pred = hr_pred * SV_pred / 60.0  # L/s

            x_k, P_k = self.ekf_update(x_pred, P_pred, z_k, Q_pred, hr_pred, vo2_pred, R_k) # at x_{t|t-1}
            x_i, P_i = x_k, P_k # at x_{t|t}

            VO2_estimate, VCO2_estimate = self.calc_ventilation_ekf(x_i[-1], x_i[0], x_i[1])
            vo2co2_est_list.append([VO2_estimate, VCO2_estimate])

            x_estimates.append(x_i)
            P_estimates.append(P_i)

        return np.array(x_estimates), P_estimates, vo2co2_est_list

class EKFObservability(EKFModel):
    def construct_observability_matrix(self):
        # ---------- 1) Symbols ----------
        # States
        x1, x2, x3, x4, x5 = sp.symbols('x1 x2 x3 x4 x5', real=True)
        x = sp.Matrix([x1, x2, x3, x4, x5])

        # Known input u(t) (beats per second) – treated as parameter at each instant
        # I cannot overwrite the symbol definition, e.g., u = u / 60
        u = sp.symbols('u', real=True)  # heart rate in beats per second

        # Delayed quantities treated as known exogenous signals
        CaO2_delay, x2_delay = sp.symbols('CaO2_delay x2_delay', real=True)

        # ---------- 2) Helper sub-expressions ----------
        # Cardiac output derivation from HR*SV, where HR = u
        VO2 = (760 - 47) / 760 * 273 / 310 * x5 * (0.21 - x1 / (760 - 47))  # L/s (constants baked)
        g = 60.0 * VO2
        eps = sp.Float('8e-4')  # VO2_eps = 0.0008
        g_pos = sp.sqrt(g ** 2 + eps ** 2)  # >= eps, smooth, no piecewise
        v_raw = 0.02 * sp.log(g_pos) + 0.08975
        # soft clip to [0.06, 0.15] using a tanh "squash"
        mid, half = 0.105, 0.045   # midpoint and half-range
        k = sp.Float(20.0)         # sharpness; 10–40 works well
        v = mid + half * sp.tanh(k * (v_raw - mid))

        # Dissociation curves
        Ce_O2 = self.const_K2 * (1 - sp.exp(-self.const_K3 * x1)) ** 2
        Ce_CO2 = self.const_K4 * x2

        # Shunt equations
        Ca_O2 = (1 - self.const_s) * Ce_O2 + self.const_s * x3
        Ca_CO2 = (1 - self.const_s) * Ce_CO2 + self.const_s * x4

        # Metabolic rates (steady-exchange assumption)
        MP_O2 = - (760 - 47) / 760 * 273 / 310  * x5 * (0.21 - x1 / (760 - 47))
        MP_CO2 = - (760 - 47) / 760 * 273 / 310 * x5 * (0.0004 - x2 / (760 - 47))

        # ---------- 3) Dynamics f(x,u) ----------
        f1 = (x5 * (self.const_P_I_O2 - x1) + self.const_lambda * u * v * (1 - self.const_s) * (x3 - Ce_O2)) / self.const_V_A
        f2 = (x5 * (self.const_P_I_CO2 - x2) + self.const_lambda * u * v * (1 - self.const_s) * (x4 - Ce_CO2)) / self.const_V_A
        f3 = (u * v * (Ca_O2 - x3) + MP_O2) / self.const_V_T
        f4 = (u * v * (Ca_CO2 - x4) + MP_CO2) / self.const_V_T
        # Delay handled as known exogenous inputs here; if you need full rigor, augment the state
        f5 = (-self.const_GPO2 * CaO2_delay + self.const_GPCO2 * x2_delay + self.const_K1 - x5) / self.const_tau

        f = sp.Matrix([f1, f2, f3, f4, f5])

        # ---------- 4) Measurement model h(x,u) ----------
        # z = [ z1; z2 ] with your definition
        h1 = u * v * (1 - self.const_s) * (Ce_O2 - x3)
        h2 = u * v * (1 - self.const_s) * (Ce_CO2 - x4)
        h = sp.Matrix([h1, h2])  # n_y = 2

        # ---------- 5) Lie derivative utility ----------
        def lie(h_vec, f_vec, x_vars):
            """L_f h = (∂h/∂x) f."""
            Jh = h_vec.jacobian(x_vars)  # (n_y x n_x)
            return Jh * f_vec  # (n_y x 1)

        # ---------- 6) Build Hermann–Krener observability matrix up to order q ----------
        nx = 5
        q = nx - 1  # up to 4 for 5 states

        blocks = []
        h_k = h
        for k in range(q + 1):
            # stack ∂(L_f^k h)/∂x
            # Jk = sp.simplify(h_k.jacobian(x)) # takes a lot of time
            blocks.append(h_k.jacobian(x))
            # prepare next L_f h
            if k < q:
                h_k = lie(h_k, f, x)

        O_HK = sp.Matrix.vstack(*blocks)  # ((q+1)*ny) x nx  => 10 x 5

        # ---------- 7) Lambdify for fast numeric evaluation ----------
        # ---------- for numerical SVD along the EKF trajectory -------
        # Collect parameters you want to feed numerically (choose what you actually vary)
        params = (u, CaO2_delay, x2_delay)

        O_HK_fun = sp.lambdify((x1, x2, x3, x4, x5, *params), O_HK, 'numpy')
        return O_HK_fun

    # === EKF main loop ===
    # === to numerically compute the observability rank along the EKF trajectory ===
    def analyze_observability(self, x0, P0, u_series, MP_O2CO2_series, t_series, N_k, R_k):
        x_estimates = [x0]
        P_estimates = [P0]
        x_i = x0
        P_i = P0
        vo2co2_est_list = []
        vo2_est_smoothed_list = []
        SV_est_smoothed_list = []

        MP_O2_series, MP_CO2_series = MP_O2CO2_series

        # initialise values
        VO2_estimate, VCO2_estimate = self.calc_ventilation_ekf(x_i[-1], x_i[0], x_i[1])
        vo2co2_est_list.append([VO2_estimate, VCO2_estimate])
        vo2_est_smoothed_list.append(VO2_estimate)
        self.smoothed_VO2_values.append(VO2_estimate)

        sv_raw = self.VO2_to_SV(VO2_estimate)
        SV_est_smoothed_list.append(sv_raw)

        observability_dict = defaultdict(list)
        # numerically analyse observability based on Lie-derivative
        print("start constructing the observability matrix O_HK symbolically")
        O_HK_fun = self.construct_observability_matrix()
        print("finish constructing the observability matrix O_HK symbolically")

        for k in range(1, len(t_series)):
            # integrated value at time step k, starting from 1
            # integration starts from time step 0
            t_span = (t_series[k - 1], t_series[k])
            delta_t = t_span[1] - t_span[0]
            i = k - 1  # inputs/index at start of interval

            hr_i = u_series[i]
            # raw VO2 (L/s): use previous estimate or a prior at k=0
            vo2_raw = vo2co2_est_list[i][0]
            self.raw_VO2_values.append(vo2_raw)
            # 1) causal EMA on VO2 (no window needed)
            tau = 10.0  # seconds (try 5–15)
            alpha = np.exp(-delta_t / tau)  # dt = 1 s
            vo2_eps = 0.0008  # vo2 floor value
            if i > 0:
                vo2_i = alpha * vo2_est_smoothed_list[i - 1] + (1 - alpha) * max(vo2_raw, vo2_eps)
            else:
                vo2_i = max(vo2_raw, vo2_eps)
            vo2_est_smoothed_list.append(vo2_i)
            self.smoothed_VO2_values.append(vo2_i)
            # 2) SV from smoothed VO2 (convert to L/min for the SV law)
            # we use VO2 estimated from previous time step to calculate the stroke volume
            # in this case, the stroke volume can be treated as a constant (thanks to the delayed term) and serves
            # as the external input to our state-space model.
            sv_raw = self.VO2_to_SV(vo2_i)  # in L/beat
            self.raw_SV_values.append(sv_raw)
            # 3) inertia + physiological bounds on SV
            beta = 0.1  # 0.05–0.2 works well
            if i > 0:
                SV_i = SV_est_smoothed_list[i - 1] + beta * (sv_raw - SV_est_smoothed_list[i - 1])
            else:
                SV_i = sv_raw
            SV_i = np.clip(SV_i, 0.06, 0.15)  # L/beat
            SV_est_smoothed_list.append(SV_i)
            self.smoothed_SV_values.append(SV_i)
            # 4) form Q (keep delay consistent; here both HR & SV are delayed)
            Q_i = hr_i * SV_i / 60.0  # L/s
            self.CO_values.append(Q_i)

            # Euler's Method
            # predict from i → i+1
            x_pred = x_i + delta_t * self.ode_rhs(x_i, Q_i)
            x_pred[4] = max(x_pred[4], 0.0)  # ensure V_dot_A is non-negative
            F = self.compute_F_state_transition_matrix(x_i, Q_i, hr_i, vo2_i,
                                                       delta_t)  # state-transition matrix at x_{t-1|t-1}
            P_pred = F @ P_i @ F.T + N_k

            # It is important that the MP_CO2_series should be negative/flipped values due to the measurement function's definition
            z_k = np.array([MP_O2_series[i + 1], -MP_CO2_series[i + 1]])
            # derive Q at time step t|t-1 based on predicted state X_{t|t-1}
            vo2_pred, _ = self.calc_ventilation_ekf(x_pred[-1], x_pred[0], x_pred[1])
            vo2_pred = alpha * vo2_est_smoothed_list[i] + (1 - alpha) * max(vo2_pred, vo2_eps)
            SV_pred = self.VO2_to_SV(vo2_pred)  # in L/beat
            SV_pred = SV_est_smoothed_list[i] + beta * (SV_pred - SV_est_smoothed_list[i])
            SV_pred = np.clip(SV_pred, 0.06, 0.15)  # L/beat
            hr_pred = u_series[i + 1]
            Q_pred = hr_pred * SV_pred / 60.0  # L/s

            # Evaluated at the predicted state x_{t|t-1}, u_{t} instead of u_{t-1}
            num = dict(u=hr_pred / 60, CaO2_delay=self.C_a_O2_delayed, x2_delay=self.P_a_CO2_delayed)
            # One evaluation:
            O_HK = O_HK_fun(x_pred[0], x_pred[1], x_pred[2], x_pred[3], x_pred[4],
                            num['u'], num['CaO2_delay'], num['x2_delay'])

            # Rank/conditioning:
            s = np.linalg.svd(np.asarray(O_HK, dtype=float), compute_uv=False)
            rank = (s > (max(O_HK.shape) * np.finfo(float).eps * s.max())).sum()
            cond = s[0] / (s[-1] if s[-1] > 0 else np.inf)
            print("Rank:", rank, "  min singular value:", s[-1], "  cond:", cond)
            observability_dict["Rank"].append(rank)
            observability_dict["Singular_Values"].append(s)
            observability_dict["Condition_Number"].append(cond)

            # ==========================================================
            # NEW: Per-state observability strength (column-wise norms)
            # ==========================================================
            # Convert to float array
            O = np.asarray(O_HK, dtype=float)
            # Optionally scale states to comparable ranges (recommended!)
            # Example if you know typical magnitudes:
            scale = np.array([100, 40, 0.144, 0.610, (4.28 / 60)])
            O = O @ np.diag(scale)
            state_strengths = np.linalg.norm(O, axis=0)  # column 2-norm per state
            # Normalize for readability (optional)
            state_strengths_norm = state_strengths / state_strengths.max()
            observability_dict["State_Strengths"].append(state_strengths_norm)
            print("Per-state observability strength (normalized):")
            for i, val in enumerate(state_strengths_norm):
                print(f"  state {i}: {val:.4f}")

            # Kalman Gain Correction
            x_k, P_k = self.ekf_update(x_pred, P_pred, z_k, Q_pred, hr_pred, vo2_pred, R_k)  # at x_{t|t-1}
            x_i, P_i = x_k, P_k  # at x_{t|t}

            VO2_estimate, VCO2_estimate = self.calc_ventilation_ekf(x_i[-1], x_i[0], x_i[1])
            vo2co2_est_list.append([VO2_estimate, VCO2_estimate])

            x_estimates.append(x_i)
            P_estimates.append(P_i)

        return observability_dict

# for Augmented State-Space
# does not work well -- suspended here for future usage

class AugEKFModel(EKFModel):
    def store_intermediate(self, t):
        # Store intermediate values in separate lists
        self.muscle_mu_values = np.append(self.muscle_mu_values, self.muscle_mu)
        self.P_a_O2_values = np.append(self.P_a_O2_values, self.P_a_O2)
        self.P_a_CO2_values = np.append(self.P_a_CO2_values, self.P_a_CO2)
        self.V_dot_A_values = np.append(self.V_dot_A_values, self.V_dot_A)
        self.C_a_O2_values = np.append(self.C_a_O2_values, self.C_a_O2)
        self.C_a_CO2_values = np.append(self.C_a_CO2_values, self.C_a_CO2)
        self.C_v_O2_values = np.append(self.C_v_O2_values, self.C_v_O2)
        self.C_v_CO2_values = np.append(self.C_v_CO2_values, self.C_v_CO2)
        self.C_e_O2_values = np.append(self.C_e_O2_values, self.C_e_O2)
        self.C_e_CO2_values = np.append(self.C_e_CO2_values, self.C_e_CO2)
        self.MPO2_values = np.append(self.MPO2_values, self.MP_O2)
        self.MPCO2_values = np.append(self.MPCO2_values, self.MP_CO2)
        self.t_values = np.append(self.t_values, t)

    def ode_rhs(self, x, Q_t, t):
        x1, x2, x3, x4, x5, x6 = x  # unpack state: P_a_O2, P_a_CO2, C_v_O2, C_v_CO2, V_dot_A, mu

        # μ is modeled as quasi-static parameter => dμ/dt = 0 --> zero-mean white process noise
        # the uncertanty part is modeled with process noise covariance
        dx6 = 0.0

        Ce_O2 = self.dissociation_curve_O2(x1)
        Ce_CO2 = self.dissociation_curve_CO2(x2)

        Ca_O2 = (1 - self.const_s) * Ce_O2 + self.const_s * x3
        Ca_CO2 = (1 - self.const_s) * Ce_CO2 + self.const_s * x4

        dx1 = (x5 * (self.const_P_I_O2 - x1) + self.const_lambda * Q_t * (1 - self.const_s) * (x3 - Ce_O2)) / self.const_V_A
        dx2 = (x5 * (self.const_P_I_CO2 - x2) + self.const_lambda * Q_t * (1 - self.const_s) * (x4 - Ce_CO2)) / self.const_V_A
        VO2, VCO2 = self.calc_ventilation_ekf(x5, x1, x2)
        dx3 = (Q_t * (Ca_O2 - x3)) / self.const_V_T - (760 - 47) / self.const_lambda * VO2
        dx4 = (Q_t * (Ca_CO2 - x4)) / self.const_V_T - (760 - 47) / self.const_lambda * (-VO2)

        if self.controller_mode == 'delayed':
            # Respiratory controller updated (with time delay)
            T = self.const_KT / Q_t
            T = int(np.round(T))
            if len(self.t_values) > 0:
                if T > self.t_values[-1] + 1:
                    C_a_O2_delayed = Ca_O2
                    P_a_CO2_delayed = x2
                else:
                    C_a_O2_delayed = self.C_a_O2_values[-1 - T + 2]
                    P_a_CO2_delayed = self.P_a_CO2_values[-1 - T + 2]
            else:
                C_a_O2_delayed = Ca_O2
                P_a_CO2_delayed = x2

            dx5 = (-self.const_GPO2 * C_a_O2_delayed +
                   self.const_GPCO2 * P_a_CO2_delayed + self.const_K1 - x5) / self.const_tau
        else:
            # original respiratory controller
            dx5 = -self.const_GPO2 * Ca_O2 + self.const_GPCO2 * x2 + self.const_K1

        # Remco's implementation:
        # Physically: Concentrations can’t be negative — this is a hard physical constraint.
        # Numerically: Prevents the solver from diverging or producing biologically invalid states.
        # Mathematically: You're not altering the dynamics arbitrarily, only truncating them to prevent unphysical values.
        # Stability: Helps avoid overflow or instability in equations like the oxygen dissociation curve.
        if x3 + dx3 < 0:
            dx3 = -x3
        if x4 + dx4 < 0:
            dx4 = -x4

        # Store values
        self.P_a_O2 = x1
        self.P_a_CO2 = x2
        self.V_dot_A = x5
        self.C_a_O2 = Ca_O2
        self.C_a_CO2 = Ca_CO2
        self.C_v_O2 = x3
        self.C_v_CO2 = x4
        self.C_e_O2 = Ce_O2
        self.C_e_CO2 = Ce_CO2
        self.C_a_O2_delayed = C_a_O2_delayed
        self.P_a_CO2_delayed = P_a_CO2_delayed
        self.muscle_mu = x6
        self.store_intermediate(t)

        return np.array([dx1, dx2, dx3, dx4, dx5, dx6])

    def compute_f_jacobian_chainrule_items(self, x, hr_t, vo2_t):
        x1, x2, x3, x4, x5, x6 = x

        Ce_O2 = self.dissociation_curve_O2(x1)
        Ce_CO2 = self.dissociation_curve_CO2(x2)

        Ca_O2 = (1 - self.const_s) * Ce_O2 + self.const_s * x3
        Ca_CO2 = (1 - self.const_s) * Ce_CO2 + self.const_s * x4

        dfdu = np.zeros((6, 1))
        dudx = np.zeros((1, 6))

        dfdu[0, 0] = self.const_lambda * (1 - self.const_s) / self.const_V_A * (x3 - Ce_O2)
        dfdu[1, 0] = self.const_lambda * (1 - self.const_s) / self.const_V_A * (x4 - Ce_CO2)
        dfdu[2, 0] = (Ca_O2 - x3) / self.const_V_T
        dfdu[3, 0] = (Ca_CO2 - x4) / self.const_V_T

        dudx[0, 0] = -hr_t / 60 * 0.02 * x5 / ((760 - 47) * vo2_t)
        dudx[0, 4] = hr_t / 60 * 0.02 * (0.21 - x1 / (760 - 47)) / vo2_t

        return dfdu @ dudx

    def compute_f_jacobian(self, x, Q_t):
        x1, x2, x3, x4, x5, x6 = x

        dCe_O2 = self.d_dissociation_curve_O2(x1)
        # dCe_CO2 = self.const_K4
        dCe_CO2 = self.d_dissociation_curve_CO2(x2)

        F = np.zeros((6, 6))
        # ∂f1/∂x1
        F[0, 0] = (-x5 - self.const_lambda * Q_t * (1 - self.const_s) * dCe_O2) / self.const_V_A
        # ∂f1/∂x3
        F[0, 2] = self.const_lambda * Q_t * (1 - self.const_s) / self.const_V_A
        # ∂f1/∂x5
        F[0, 4] = (self.const_P_I_O2 - x1) / self.const_V_A

        # ∂f2/∂x2
        # updated
        # F[1, 1] = (-x5 - self.const_lambda * Q_t * (1 - self.const_s) * dCe_O2) / self.const_V_A
        F[1, 1] = (-x5 - self.const_lambda * Q_t * (1 - self.const_s) * dCe_CO2) / self.const_V_A
        # ∂f2/∂x4
        F[1, 3] = self.const_lambda * Q_t * (1 - self.const_s) / self.const_V_A
        # ∂f2/∂x5
        F[1, 4] = (self.const_P_I_CO2 - x2) / self.const_V_A

        # ∂f3/∂x1
        # updated
        # F[2, 0] = Q_t * (1 - self.const_s) * dCe_O2 / self.const_V_T
        F[2, 0] = (Q_t * (1 - self.const_s) * dCe_O2 + x5 / self.const_lambda) / self.const_V_T
        # ∂f3/∂x3
        F[2, 2] = (Q_t / self.const_V_T) * (self.const_s - 1)
        # ∂f3/∂x5
        # added
        F[2, 4] = (- (760 - 47) * 0.21 / self.const_lambda + x1 / self.const_lambda) / self.const_V_T

        # ∂f4/∂x2
        # updated
        # F[3, 1] = Q_t * (1 - self.const_s) * dCe_CO2 / self.const_V_T
        F[3, 1] = (Q_t * (1 - self.const_s) * dCe_CO2 + x5 / self.const_lambda) / self.const_V_T
        # ∂f4/∂x4
        F[3, 3] = (Q_t / self.const_V_T) * (self.const_s - 1)
        # ∂f4/∂x5
        # added
        F[3, 4] = (- (760 - 47) * 0.0006 / self.const_lambda + x2 / self.const_lambda) / self.const_V_T

        if self.controller_mode == 'delayed':
            # With time delay: treat C_aO2(t-T), x2(t-T) as exogenous (no dependence on current x)
            # ∂f5/∂x5
            F[4, 4] = - 1 / self.const_tau
        else:
            # For respiratory controller without time delay
            # No time delay: controller depends on current CaO2(x1,x3) and x2
            F[4, 0] = -(self.const_GPO2 / self.const_tau) * (1 - self.const_s) * dCe_O2  # ∂f5/∂x1
            F[4, 1] = (self.const_GPCO2 / self.const_tau)  # ∂f5/∂x2
            F[4, 2] = -(self.const_GPO2 / self.const_tau) * self.const_s  # ∂f5/∂x3
            F[4, 4] = -1.0 / self.const_tau  # ∂f5/∂x5

        return F

    def compute_F_state_transition_matrix(self, x, Q_t, hr_t, vo2_t, delta_t):
        # continuous-time dynamic Jacobian
        f_jacobian = self.compute_f_jacobian(x, Q_t) + self.compute_f_jacobian_chainrule_items(x, hr_t, vo2_t)
        # state-transition matrix for discrete-time
        F_jacobian = f_jacobian * delta_t + np.eye(6)

        return F_jacobian

    def measurement_model(self, x, Q_t):
        x1, x2, x3, x4, _, x6 = x
        Ce_O2 = self.dissociation_curve_O2(x1)
        Ce_CO2 = self.dissociation_curve_CO2(x2)

        base_O2 = x6 * Q_t * (1 - self.const_s) * (Ce_O2 - x3)
        base_CO2 = x6 * Q_t * (1 - self.const_s) * (Ce_CO2 - x4)
        return np.array([base_O2, base_CO2])
        # return base_O2

    def compute_h_jacobian_chainrule_items(self, x, hr_t, vo2_t):
        x1, x2, x3, x4, x5, x6 = x

        Ce_O2 = self.dissociation_curve_O2(x1)
        Ce_CO2 = self.dissociation_curve_CO2(x2)

        dhdu = np.zeros((2, 1))
        dudx = np.zeros((1, 6))

        dhdu[0, 0] = x6 * (1 - self.const_s) * (Ce_O2 - x3)
        dhdu[1, 0] = x6 * (1 - self.const_s) * (Ce_CO2 - x4)

        dudx[0, 0] = -hr_t / 60 * 0.02 * x5 / ((760 - 47) * vo2_t)
        dudx[0, 4] = hr_t / 60 * 0.02 * (0.21 - x1 / (760 - 47)) / vo2_t

        return dhdu @ dudx

    def compute_H_jacobian(self, x, Q_t, hr_t, vo2_t):
        x1, x2, x3, x4, x5, x6 = x

        dCeO2_dx1 = self.d_dissociation_curve_O2(x1)
        dCeCO2_dx2 = self.d_dissociation_curve_CO2(x2)
        CeO2 = self.dissociation_curve_O2(x1)
        CeCO2 = self.dissociation_curve_CO2(x2)
        H = np.zeros((2, 6))
        factor = Q_t * (1 - self.const_s)
        # ∂z1/∂x1 and ∂z1/∂x3
        H[0, 0] = x6 * factor * dCeO2_dx1
        H[0, 2] = -x6 * factor
        # ∂z1/∂x6
        H[0, 5] = factor * (CeO2 - x3)

        # ∂z2/∂x2 and ∂z2/∂x4
        H[1, 1] = x6 * factor * dCeCO2_dx2
        H[1, 3] = -x6 * factor
        # ∂z2/∂x6
        H[1, 5] = factor * (CeCO2 - x4)

        H = H + self.compute_h_jacobian_chainrule_items(x, hr_t, vo2_t)

        return H

    def update_parameters(self):
        hr = self.hr
        SV = self.SV

        # Change Q
        if self.Q_mode == 'dynamic':
            self.Q = hr * SV / 60  # Convert to seconds
        else:
            self.Q = np.zeros(len(hr))
            for t_index in range(len(self.activity_labels)):
                if self.activity_labels[t_index] in self.Cycling_list:
                    self.Q[t_index] = self.Q_exercise
                elif self.activity_labels[t_index] in self.LWBM_list:
                    self.Q[t_index] = self.Q_rest
                elif self.activity_labels[t_index] in self.HWBM_list:
                    self.Q[t_index] = self.Q_exercise
                elif self.activity_labels[t_index] in self.Walking_list:
                    self.Q[t_index] = self.Q_exercise
                else:
                    self.Q[t_index] = self.Q_rest

        # Change MP_O2 and MP_CO2
        MP_O2, MP_CO2 = self.velocity_to_MP()
        self.MP_O2_values = MP_O2
        self.MP_CO2_values = MP_CO2

    def velocity_to_MP(self):

        approach_nr = self.approach_nr

        # Only pelvis
        if approach_nr == 1:
            kinetic_energy = 0.5 * self.mass * (self.velocity ** 2)

        # Pelvis, both legs
        # This approach tends to overestimate treadmill 3km and treadmill 5km, while underestimate cycling
        elif approach_nr == 2:
            mass = self.mass
            mass_torso = mass * 0.6664 if self.sex == 'male' else mass * 0.6314
            mass_leg = mass * 0.1668 if self.sex == 'male' else mass * 0.1843
            KE_torso = 0.5 * mass_torso * (self.velocity_DOT7 ** 2)
            KE_right_leg = 0.5 * mass_leg * (self.velocity_DOT8 ** 2)
            KE_left_leg = 0.5 * mass_leg * (self.velocity_DOT9 ** 2)

            kinetic_energy = KE_torso + KE_left_leg + KE_right_leg

        # Pelvis, both legs, with gyrations included
        elif approach_nr == 3:
            mass = self.mass
            height = self.height / 100 # meters
            mass_torso = mass * 0.8332 if self.sex == 'male' else mass * 0.8157
            mass_leg = mass * 0.1668 if self.sex == 'male' else mass * 0.1843
            # Segment length
            length_torso = 0.3 * height
            length_leg = 0.45 * height
            # Moments of inertia
            I_torso = (1 / 12) * mass_torso * (length_torso ** 2)
            I_leg = (1 / 3) * mass_leg * (length_leg ** 2)
            # Translational KE
            KE_torso = 0.5 * mass_torso * (self.velocity_DOT7 ** 2)
            KE_right_leg = 0.5 * mass_leg * (self.velocity_DOT8 ** 2)
            KE_left_leg = 0.5 * mass_leg * (self.velocity_DOT9 ** 2)
            # Rotational KE
            KE_rot_torso = 0.5 * I_torso * (self.gyration_DOT7 ** 2)
            KE_rot_right_leg = 0.5 * I_leg * (self.gyration_DOT8 ** 2)
            KE_rot_left_leg = 0.5 * I_leg * (self.gyration_DOT9 ** 2)

            kinetic_energy = ( KE_torso + KE_right_leg + KE_left_leg +
                               KE_rot_torso + KE_rot_right_leg + KE_rot_left_leg)

        elif self.approach_nr == 4:
            mass = self.mass
            mass_torso = mass * 0.5664 if self.sex == 'male' else mass * 0.5314
            mass_leg = mass * 0.2168 if self.sex == 'male' else mass * 0.2343
            # mass_wrist = mass * 0.022 / 2 if self.sex == 'male' else mass * 0.019 / 2
            # mass_foot = 0.0145 * mass if self.sex == 'male' else 0.0132 * mass
            KE_torso = 0.5 * mass_torso * (self.velocity_DOT7 ** 2)
            KE_right_leg = 0.5 * mass_leg * (self.velocity_DOT8 ** 2)
            KE_left_leg = 0.5 * mass_leg * (self.velocity_DOT9 ** 2)
            # KE_left_foot = 0.5 * mass_foot * (self.velocity_DOT10 ** 2)
            # KE_right_wrist = 0.5 * mass_wrist * (self.velocity_DOT6 ** 2)
            # KE_left_wrist = 0.5 * mass_wrist * (self.velocity_DOT5 ** 2)

            kinetic_energy = KE_torso + KE_right_leg + KE_left_leg

        energy = kinetic_energy / 1000
        energy = np.array([0 if a < 0 else a for a in energy])
        MP_O2 = energy / 19.6

        if self.RQ_mode == 'constant':
            MP_CO2 = self.const_RQ * MP_O2
        elif self.RQ_mode == 'adaptive':
            RQ_list = np.array([0.7] * len(energy))
            for t_index in range(len(self.activity_labels)):
                if self.activity_labels[t_index] in self.Cycling_list:
                    RQ_list[t_index] = 0.8
                elif self.activity_labels[t_index] in self.LWBM_list:
                    RQ_list[t_index] = 0.7
                elif self.activity_labels[t_index] in self.HWBM_list:
                    RQ_list[t_index] = 0.8
                elif self.activity_labels[t_index] in self.Walking_list:
                    RQ_list[t_index] = 0.8
            MP_CO2 = RQ_list * MP_O2

        return MP_O2, MP_CO2

    def ekf_update(self, t_step, x_pred, P_pred, z_meas, Q_t, hr_t, vo2_t, R_k):
        H = self.compute_H_jacobian(x_pred, Q_t, hr_t, vo2_t) # x_{t|t-1}, Q_{t} or Q_{t|t-1}

        z_pred = self.measurement_model(x_pred, Q_t) # x_{t|t-1}
        y = z_meas - z_pred
        # y = np.array([[z_meas - z_pred]])
        S = H @ P_pred @ H.T + R_k
        K = P_pred @ H.T @ np.linalg.inv(S)

        if t_step <= 100:
            K[5, :] = 0

        x_upd = x_pred + K @ y
        # x_upd = x_pred + (K @ y).flatten()

        # P_upd = (np.eye(len(x_pred)) - K @ H) @ P_pre

        # Joseph's form, numerically safer
        I = np.eye(len(x_pred))
        P_upd = (I - K @ H) @ P_pred @ (I - K @ H).T + K @ R_k @ K.T
        # optional: enforce symmetry against roundoff
        P_upd = 0.5 * (P_upd + P_upd.T)

        innovation_norm = np.linalg.norm(y)  # ||y||
        expected_std = np.sqrt(np.trace(S))  # ~ RMS expected residual

        return x_upd, P_upd, innovation_norm, expected_std

    # === EKF main loop ===
    def run_ekf(self, x0, P0, u_series, MP_O2CO2_series, t_series, N_k, R_k):
        x_estimates = [x0]
        P_estimates = [P0]
        x_i = x0
        P_i = P0
        vo2co2_est_list = []
        vo2_est_smoothed_list = []
        SV_est_smoothed_list = []

        MP_O2_series, MP_CO2_series = MP_O2CO2_series

        # initialise values
        VO2_estimate, VCO2_estimate = self.calc_ventilation_ekf(x_i[4], x_i[0], x_i[1])
        vo2co2_est_list.append([VO2_estimate, VCO2_estimate])
        vo2_est_smoothed_list.append(VO2_estimate)
        self.smoothed_VO2_values.append(VO2_estimate)

        sv_raw = self.VO2_to_SV(VO2_estimate)
        SV_est_smoothed_list.append(sv_raw)

        ratio_list = []

        for k in range(1, len(t_series)):
            # integrated value at time step k, starting from 1
            # integration starts from time step 0
            t_span = (t_series[k - 1], t_series[k])
            delta_t = t_span[1] - t_span[0]
            i = k - 1  # inputs/index at start of interval

            hr_i = u_series[i]
            # raw VO2 (L/s): use previous estimate or a prior at k=0
            vo2_raw = vo2co2_est_list[i][0]
            self.raw_VO2_values.append(vo2_raw)
            # 1) causal EMA on VO2 (no window needed)
            tau = 10.0  # seconds (try 5–15)
            alpha = np.exp(-delta_t / tau)  # dt = 1 s
            vo2_eps = 0.0008  # vo2 floor value
            if i > 0:
                vo2_i = alpha * vo2_est_smoothed_list[i - 1] + (1 - alpha) * max(vo2_raw, vo2_eps)
            else:
                vo2_i = max(vo2_raw, vo2_eps)
            vo2_est_smoothed_list.append(vo2_i)
            self.smoothed_VO2_values.append(vo2_i)
            # 2) SV from smoothed VO2 (convert to L/min for the SV law)
            # we use VO2 estimated from previous time step to calculate the stroke volume
            # in this case, the stroke volume can be treated as a constant (thanks to the delayed term) and serves
            # as the external input to our state-space model.
            sv_raw = self.VO2_to_SV(vo2_i)  # in L/beat
            self.raw_SV_values.append(sv_raw)
            # 3) inertia + physiological bounds on SV
            beta = 0.1  # 0.05–0.2 works well
            if i > 0:
                SV_i = SV_est_smoothed_list[i - 1] + beta * (sv_raw - SV_est_smoothed_list[i - 1])
            else:
                SV_i = sv_raw
            SV_i = np.clip(SV_i, 0.06, 0.15)  # L/beat
            SV_est_smoothed_list.append(SV_i)
            self.smoothed_SV_values.append(SV_i)
            # 4) form Q (keep delay consistent; here both HR & SV are delayed)
            Q_i = hr_i * SV_i / 60.0  # L/s
            self.CO_values.append(Q_i)

            # Euler's Method
            # predict from i → i+1
            x_pred = x_i + delta_t * self.ode_rhs(x_i, Q_i, i)
            x_pred[4] = max(x_pred[4], 0.0)  # ensure V_dot_A is non-negative
            #x_pred[5] = np.clip(x_pred[5], 2, 3) # clip muscle_mu value range

            F = self.compute_F_state_transition_matrix(x_i, Q_i, hr_i, vo2_i,
                                                       delta_t)  # state-transition matrix at x_{t-1|t-1}
            P_pred = F @ P_i @ F.T + N_k

            # It is important that the MP_CO2_series should be negative/flipped values due to the measurement function's definition
            z_k = np.array([MP_O2_series[i + 1], -MP_CO2_series[i + 1]])
            # derive Q at time step t|t-1 based on predicted state X_{t|t-1}
            vo2_pred, _ = self.calc_ventilation_ekf(x_pred[4], x_pred[0], x_pred[1])
            vo2_pred = alpha * vo2_est_smoothed_list[i] + (1 - alpha) * max(vo2_pred, vo2_eps)
            SV_pred = self.VO2_to_SV(vo2_pred)  # in L/beat
            SV_pred = SV_est_smoothed_list[i] + beta * (SV_pred - SV_est_smoothed_list[i])
            SV_pred = np.clip(SV_pred, 0.06, 0.15)  # L/beat
            hr_pred = u_series[i + 1]
            Q_pred = hr_pred * SV_pred / 60.0  # L/s

            x_k, P_k, inno_norm, expected_std = self.ekf_update(k, x_pred, P_pred, z_k, Q_pred, hr_pred, vo2_pred, R_k)  # at x_{t|t-1}
            x_i, P_i = x_k, P_k  # at x_{t|t}

            ratio = inno_norm / expected_std
            ratio_list.append(ratio)

            # numerically analyse observability based on Lie-derivative

            VO2_estimate, VCO2_estimate = self.calc_ventilation_ekf(x_i[4], x_i[0], x_i[1])
            vo2co2_est_list.append([VO2_estimate, VCO2_estimate])

            x_estimates.append(x_i)
            P_estimates.append(P_i)

        return np.array(x_estimates), P_estimates, vo2co2_est_list, ratio_list

class AugEKFObservability(AugEKFModel):
    def construct_observability_matrix(self):
        # ---------- 1) Symbols ----------
        # States
        x1, x2, x3, x4, x5, x6 = sp.symbols('x1 x2 x3 x4 x5 x6', real=True)
        x = sp.Matrix([x1, x2, x3, x4, x5, x6])

        # Known input u(t) (beats per second) – treated as parameter at each instant
        # I cannot overwrite the symbol definition, e.g., u = u / 60
        u = sp.symbols('u', real=True)  # heart rate in beats per second

        # Delayed quantities treated as known exogenous signals
        CaO2_delay, x2_delay = sp.symbols('CaO2_delay x2_delay', real=True)

        # ---------- 2) Helper sub-expressions ----------
        # Cardiac output derivation from HR*SV, where HR = u
        VO2 = x5 * (0.21 - x1 / (760 - 47))  # L/s (constants baked)
        g = 60.0 * VO2
        eps = sp.Float('8e-4')  # VO2_eps = 0.0008
        g_pos = sp.sqrt(g ** 2 + eps ** 2)  # >= eps, smooth, no piecewise
        v_raw = 0.02 * sp.log(g_pos) + 0.08975
        # soft clip to [0.06, 0.15] using a tanh "squash"
        mid, half = 0.105, 0.045   # midpoint and half-range
        k = sp.Float(20.0)         # sharpness; 10–40 works well
        v = mid + half * sp.tanh(k * (v_raw - mid))

        # Dissociation curves
        Ce_O2 = self.const_K2 * (1 - sp.exp(-self.const_K3 * x1)) ** 2
        Ce_CO2 = self.const_K4 * x2

        # Shunt equations
        Ca_O2 = (1 - self.const_s) * Ce_O2 + self.const_s * x3
        Ca_CO2 = (1 - self.const_s) * Ce_CO2 + self.const_s * x4

        # Metabolic rates (steady-exchange assumption)
        MP_O2 = - (760 - 47) / self.const_lambda * x5 * (0.21 - x1 / (760 - 47))
        MP_CO2 = - (760 - 47) / self.const_lambda * x5 * (0.0004 - x2 / (760 - 47))

        # ---------- 3) Dynamics f(x,u) ----------
        f1 = (x5 * (self.const_P_I_O2 - x1) + self.const_lambda * u * v * (1 - self.const_s) * (x3 - Ce_O2)) / self.const_V_A
        f2 = (x5 * (self.const_P_I_CO2 - x2) + self.const_lambda * u * v * (1 - self.const_s) * (x4 - Ce_CO2)) / self.const_V_A
        f3 = (u * v * (Ca_O2 - x3) + MP_O2) / self.const_V_T
        f4 = (u * v * (Ca_CO2 - x4) + MP_CO2) / self.const_V_T
        # Delay handled as known exogenous inputs here; if you need full rigor, augment the state
        f5 = (-self.const_GPO2 * CaO2_delay + self.const_GPCO2 * x2_delay + self.const_K1 - x5) / self.const_tau
        f6 = 0

        f = sp.Matrix([f1, f2, f3, f4, f5, f6])

        # ---------- 4) Measurement model h(x,u) ----------
        # z = [ z1; z2 ] with your definition
        h1 = 1/x6 * u * v * (1 - self.const_s) * (Ce_O2 - x3)
        h2 = 1/x6 * u * v * (1 - self.const_s) * (Ce_CO2 - x4)
        h = sp.Matrix([h1, h2])  # n_y = 2

        # ---------- 5) Lie derivative utility ----------
        def lie(h_vec, f_vec, x_vars):
            """L_f h = (∂h/∂x) f."""
            Jh = h_vec.jacobian(x_vars)  # (n_y x n_x)
            return Jh * f_vec  # (n_y x 1)

        # ---------- 6) Build Hermann–Krener observability matrix up to order q ----------
        nx = 6
        q = nx - 2  # up to 4 for 5 states

        blocks = []
        h_k = h
        for k in range(q + 1):
            # stack ∂(L_f^k h)/∂x
            # Jk = sp.simplify(h_k.jacobian(x)) # takes a lot of time
            blocks.append(h_k.jacobian(x))
            # prepare next L_f h
            if k < q:
                h_k = lie(h_k, f, x)

        O_HK = sp.Matrix.vstack(*blocks)  # ((q+1)*ny) x nx  => 10 x 5

        # ---------- 7) Lambdify for fast numeric evaluation ----------
        # ---------- for numerical SVD along the EKF trajectory -------
        # Collect parameters you want to feed numerically (choose what you actually vary)
        params = (u, CaO2_delay, x2_delay)

        O_HK_fun = sp.lambdify((x1, x2, x3, x4, x5, x6, *params), O_HK, 'numpy')
        return O_HK_fun

    # === EKF main loop ===
    # === to numerically compute the observability rank along the EKF trajectory ===
    def analyze_observability(self, x0, P0, u_series, MP_O2CO2_series, t_series, N_k, R_k):
        x_estimates = [x0]
        P_estimates = [P0]
        x_i = x0
        P_i = P0
        vo2co2_est_list = []
        vo2_est_smoothed_list = []
        SV_est_smoothed_list = []

        MP_O2_series, MP_CO2_series = MP_O2CO2_series

        # initialise values
        VO2_estimate, VCO2_estimate = self.calc_ventilation_ekf(x_i[4], x_i[0], x_i[1])
        vo2co2_est_list.append([VO2_estimate, VCO2_estimate])
        vo2_est_smoothed_list.append(VO2_estimate)
        self.smoothed_VO2_values.append(VO2_estimate)

        sv_raw = self.VO2_to_SV(VO2_estimate)
        SV_est_smoothed_list.append(sv_raw)

        observability_dict = defaultdict(list)
        # numerically analyse observability based on Lie-derivative
        print("start constructing the observability matrix O_HK symbolically")
        O_HK_fun = self.construct_observability_matrix()
        print("finish constructing the observability matrix O_HK symbolically")

        for k in range(1, len(t_series)):
            # integrated value at time step k, starting from 1
            # integration starts from time step 0
            t_span = (t_series[k - 1], t_series[k])
            delta_t = t_span[1] - t_span[0]
            i = k - 1  # inputs/index at start of interval

            hr_i = u_series[i]
            # raw VO2 (L/s): use previous estimate or a prior at k=0
            vo2_raw = vo2co2_est_list[i][0]
            self.raw_VO2_values.append(vo2_raw)
            # 1) causal EMA on VO2 (no window needed)
            tau = 10.0  # seconds (try 5–15)
            alpha = np.exp(-delta_t / tau)  # dt = 1 s
            vo2_eps = 0.0008  # vo2 floor value
            if i > 0:
                vo2_i = alpha * vo2_est_smoothed_list[i - 1] + (1 - alpha) * max(vo2_raw, vo2_eps)
            else:
                vo2_i = max(vo2_raw, vo2_eps)
            vo2_est_smoothed_list.append(vo2_i)
            self.smoothed_VO2_values.append(vo2_i)
            # 2) SV from smoothed VO2 (convert to L/min for the SV law)
            # we use VO2 estimated from previous time step to calculate the stroke volume
            # in this case, the stroke volume can be treated as a constant (thanks to the delayed term) and serves
            # as the external input to our state-space model.
            sv_raw = self.VO2_to_SV(vo2_i)  # in L/beat
            self.raw_SV_values.append(sv_raw)
            # 3) inertia + physiological bounds on SV
            beta = 0.1  # 0.05–0.2 works well
            if i > 0:
                SV_i = SV_est_smoothed_list[i - 1] + beta * (sv_raw - SV_est_smoothed_list[i - 1])
            else:
                SV_i = sv_raw
            SV_i = np.clip(SV_i, 0.06, 0.15)  # L/beat
            SV_est_smoothed_list.append(SV_i)
            self.smoothed_SV_values.append(SV_i)
            # 4) form Q (keep delay consistent; here both HR & SV are delayed)
            Q_i = hr_i * SV_i / 60.0  # L/s
            self.CO_values.append(Q_i)

            # Euler's Method
            # predict from i → i+1
            x_pred = x_i + delta_t * self.ode_rhs(x_i, Q_i, i)
            x_pred[4] = max(x_pred[4], 0.0)  # ensure V_dot_A is non-negative
            # x_pred[5] = np.clip(x_pred[5], 2, 3)  # clip muscle_mu value range

            F = self.compute_F_state_transition_matrix(x_i, Q_i, hr_i, vo2_i,
                                                       delta_t)  # state-transition matrix at x_{t-1|t-1}
            P_pred = F @ P_i @ F.T + N_k

            # It is important that the MP_CO2_series should be negative/flipped values due to the measurement function's definition
            z_k = np.array([MP_O2_series[i + 1], -MP_CO2_series[i + 1]]) * x_pred[5]
            # derive Q at time step t|t-1 based on predicted state X_{t|t-1}
            vo2_pred, _ = self.calc_ventilation_ekf(x_pred[4], x_pred[0], x_pred[1])
            vo2_pred = alpha * vo2_est_smoothed_list[i] + (1 - alpha) * max(vo2_pred, vo2_eps)
            SV_pred = self.VO2_to_SV(vo2_pred)  # in L/beat
            SV_pred = SV_est_smoothed_list[i] + beta * (SV_pred - SV_est_smoothed_list[i])
            SV_pred = np.clip(SV_pred, 0.06, 0.15)  # L/beat
            hr_pred = u_series[i + 1]
            Q_pred = hr_pred * SV_pred / 60.0  # L/s

            # Evaluated at the predicted state x_{t|t-1}, u_{t} instead of u_{t-1}
            num = dict(u=hr_pred / 60, CaO2_delay=self.C_a_O2_delayed, x2_delay=self.P_a_CO2_delayed)
            # One evaluation:
            O_HK = O_HK_fun(x_pred[0], x_pred[1], x_pred[2], x_pred[3], x_pred[4], x_pred[5],
                            num['u'], num['CaO2_delay'], num['x2_delay'])

            # Rank/conditioning:
            s = np.linalg.svd(np.asarray(O_HK, dtype=float), compute_uv=False)
            rank = (s > (max(O_HK.shape) * np.finfo(float).eps * s.max())).sum()
            cond = s[0] / (s[-1] if s[-1] > 0 else np.inf)
            print("Rank:", rank, "  min singular value:", s[-1], "  cond:", cond)
            observability_dict["Rank"].append(rank)
            observability_dict["Singular_Values"].append(s)
            observability_dict["Condition_Number"].append(cond)

            x_k, P_k, inno_norm, expected_std = self.ekf_update(k, x_pred, P_pred, z_k, Q_pred, hr_pred, vo2_pred,
                                                                R_k)  # at x_{t|t-1}
            x_i, P_i = x_k, P_k  # at x_{t|t}

            VO2_estimate, VCO2_estimate = self.calc_ventilation_ekf(x_i[4], x_i[0], x_i[1])
            vo2co2_est_list.append([VO2_estimate, VCO2_estimate])

            x_estimates.append(x_i)
            P_estimates.append(P_i)

        return observability_dict, vo2co2_est_list
