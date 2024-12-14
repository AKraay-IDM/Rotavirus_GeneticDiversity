"""
RotaABM class

Usage:
    import rotasim as rs
    sim = rs.Sim()
    sim.run()
"""

import numpy as np
import sciris as sc
import starsim as ss

# Define age bins and labels
age_bins = [2/12, 4/12, 6/12, 12/12, 24/12, 36/12, 48/12, 60/12, 100]
age_distribution = [0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.84]                  # needs to be changed to fit the site-specific population
age_labels = ['0-2', '2-4', '4-6', '6-12', '12-24', '24-36', '36-48', '48-60', '60+']


class RotaPeople(ss.People):
    """
    A rotavirus host
    """
    def __init__(self, *args, **kwargs):
        states = [
            ss.FloatArr('bday', default=ss.uniform(0, 5)),
            ss.Arr('immunity', dtype=object),
            ss.BoolArr('vaccine'),
            ss.Arr('infecting_pathogen', dtype=object),
            ss.FloatArr('prior_infections'),
            ss.Arr('prior_vaccinations', dtype=object),
            ss.Arr('infections_with_vaccination', dtype=object),
            ss.Arr('infections_without_vaccination', dtype=object),
            ss.BoolArr('is_immune_flag'),
            ss.FloatArr('oldest_infection', default=np.nan),
        ]
        super().__init__(*args, extra_states=states, **kwargs)
        return


    # SS:TODO
    def compute_combinations(self):

        seg_combinations = []

        # We want to only reassort the GP types
        # Assumes that antigenic segments are at the start
        for i in range(self.numAgSegments):
            availableVariants = set([])
            for j in self.infecting_pathogen:
                availableVariants.add((j.strain[i]))
            seg_combinations.append(availableVariants)

        # compute the parental strains
        parantal_strains = [j.strain[:self.numAgSegments] for j in self.infecting_pathogen]

        # Itertools product returns all possible combinations
        # We are only interested in strain combinations that are reassortants of the parental strains
        # We need to skip all existing combinations from the parents
        # Ex: (1, 1, 2, 2) and (2, 2, 1, 1) should not create (1, 1, 1, 1) as a possible reassortant if only the antigenic parts reassort

        # below block is for reassorting antigenic segments only
        all_antigenic_combinations = [i for i in itertools.product(*seg_combinations) if i not in parantal_strains]
        all_nonantigenic_combinations = [j.strain[self.numAgSegments:] for j in self.infecting_pathogen]
        all_strains = set([(i[0] + i[1]) for i in itertools.product(all_antigenic_combinations, all_nonantigenic_combinations)])
        all_pathogens = [Pathogen(self.sim, True, self.t, host = self, strain=tuple(i)) for i in all_strains]

        return all_pathogens


    def recover(self,strain_counts):
        # We will use the pathogen creation time to count the number of infections
        creation_times = set()
        for path in self.infecting_pathogen:
            strain = path.strain
            if not path.is_reassortant:
                strain_counts[strain] -= 1
                creation_times.add(path.creation_time)
                self.immunity[strain] = self.t
                self.is_immune_flag = True
                if np.isnan(self.oldest_infection):
                    self.oldest_infection = self.t
        self.prior_infections += len(creation_times)
        self.infecting_pathogen = []
        self.possibleCombinations = []

    def vaccinate(self, vaccinated_strain):
        if len(self.prior_vaccinations) == 0:
            self.prior_vaccinations.append(vaccinated_strain)
            self.vaccine = ([vaccinated_strain], self.t, 1)
        else:
            self.prior_vaccinations.append(vaccinated_strain)
            self.vaccine = ([vaccinated_strain], self.t, 2)

    def is_vaccine_immune(self, infecting_strain):
        # Effectiveness of the vaccination depends on the number of doses
        if self.vaccine[2] == 1:
            ve_i_rates = self.sim.vaccine_efficacy_i_d1
        elif self.vaccine[2] == 2:
            ve_i_rates = self.sim.vaccine_efficacy_i_d2
        else:
            raise NotImplementedError(f"Unsupported vaccine dose: {self.vaccine[2]}")

        # Vaccine strain only contains the antigenic parts
        vaccine_strain = self.vaccine[0]
        vaccine_hypothesis = self.sim.vaccine_hypothesis

        if vaccine_hypothesis == 0:
            return False
        if vaccine_hypothesis == 1:
            if infecting_strain[:self.numAgSegments] in vaccine_strain:
                if rnd.random() < ve_i_rates[PathogenMatch.HOMOTYPIC]:
                    return True
                else:
                    return False
        elif vaccine_hypothesis == 2:
            if infecting_strain[:self.numAgSegments] in vaccine_strain:
                if rnd.random() < ve_i_rates[PathogenMatch.HOMOTYPIC]:
                    return True
                else:
                    return False
            strains_match = False
            for i in range(self.numAgSegments):
                immune_genotypes = [strain[i] for strain in vaccine_strain]
                if infecting_strain[i] in immune_genotypes:
                    strains_match = True
            if strains_match:
                if rnd.random() < ve_i_rates[PathogenMatch.PARTIAL_HETERO]:
                    return True
            else:
                return False
        # used below hypothesis for the analysis in the report
        elif vaccine_hypothesis == 3:
            if infecting_strain[:self.numAgSegments] in vaccine_strain:
                if rnd.random() < ve_i_rates[PathogenMatch.HOMOTYPIC]:
                    return True
                else:
                    return False
            strains_match = False
            for i in range(self.numAgSegments):
                immune_genotypes = [strain[i] for strain in vaccine_strain]
                if infecting_strain[i] in immune_genotypes:
                    strains_match = True
            if strains_match:
                if rnd.random() < ve_i_rates[PathogenMatch.PARTIAL_HETERO]:
                    return True
            else:
                if rnd.random() < ve_i_rates[PathogenMatch.COMPLETE_HETERO]:
                    return True
                else:
                    return False
        else:
            raise NotImplementedError("Unsupported vaccine hypothesis")

    def can_variant_infect_host(self, infecting_strain, current_infections):
        numAgSegments = self.numAgSegments
        immunity_hypothesis = self.sim.immunity_hypothesis
        partial_cross_immunity_rate = self.sim.partial_cross_immunity_rate
        complete_heterotypic_immunity_rate = self.sim.complete_heterotypic_immunity_rate
        homotypic_immunity_rate = self.sim.homotypic_immunity_rate

        if self.vaccine is not None and self.is_vaccine_immune(infecting_strain):
            return False

        current_infecting_strains = (i.strain[:numAgSegments] for i in current_infections)
        if infecting_strain[:numAgSegments] in current_infecting_strains:
            return False

        def is_completely_immune():
            immune_strains = (s[:numAgSegments] for s in self.immunity.keys())
            return infecting_strain[:numAgSegments] in immune_strains

        def has_shared_genotype():
            for i in range(numAgSegments):
                immune_genotypes = (strain[i] for strain in self.immunity.keys())
                if infecting_strain[i] in immune_genotypes:
                    return True
            return False

        if immunity_hypothesis == 1:
            if is_completely_immune():
                return False
            return True

        elif immunity_hypothesis == 2:
            if has_shared_genotype():
                return False
            return True

        elif immunity_hypothesis in [3, 4, 7, 8, 9, 10]:
            if is_completely_immune():
                if immunity_hypothesis in [7, 8, 9, 10]:
                    if rnd.random() < homotypic_immunity_rate:
                        return False
                else:
                    return False

            if has_shared_genotype():
                if rnd.random() < partial_cross_immunity_rate:
                    return False
            elif immunity_hypothesis in [4, 7, 8, 9, 10]:
                if rnd.random() < complete_heterotypic_immunity_rate:
                    return False
            return True

        elif immunity_hypothesis == 5:
            immune_ptypes = (strain[1] for strain in self.immunity.keys())
            return infecting_strain[1] not in immune_ptypes

        elif immunity_hypothesis == 6:
            immune_ptypes = (strain[0] for strain in self.immunity.keys())
            return infecting_strain[0] not in immune_ptypes

        else:
            raise NotImplementedError(f"Immunity hypothesis {immunity_hypothesis} is not implemented")

    def record_infection(self, new_p):
        if len(self.prior_vaccinations) != 0:
            vaccine_strain = self.prior_vaccinations[-1]
            self.infections_with_vaccination.append((new_p, new_p.match(vaccine_strain)))
        else:
            self.infections_without_vaccination.append(new_p)

    def infect_with_pathogen(self, pathogen_in, strain_counts):
        """ This function returns a fitness value to a strain based on the hypothesis """
        fitness = pathogen_in.get_fitness()

        # e.g. fitness = 0.8 (there's a 80% chance the virus infecting a host)
        if rnd.random() > fitness:
            return False

        # Probability of getting a severe decease depends on the number of previous infections and vaccination status of the host
        severity_probability = self.sim.get_probability_of_severe(self.sim, pathogen_in, self.vaccine, self.prior_infections)
        if rnd.random() < severity_probability:
            severe = True
        else:
            severe = False

        new_p = Pathogen(self.sim, False, self.t, host = self, strain=pathogen_in.strain, is_severe=severe)
        self.infecting_pathogen.append(new_p)
        self.record_infection(new_p)

        strain_counts[new_p.strain] += 1

        return True

    def infect_with_reassortant(self, reassortant_virus):
        self.infecting_pathogen.append(reassortant_virus)



### Pathogen classes

class Pathogen:
    """
    Pathogen dynamics
    """
    def __init__(self, sim, is_reassortant, creation_time, is_severe=False, host=None, strain=None):
        self.sim = sim
        self.host = host
        self.creation_time = creation_time
        self.is_reassortant = is_reassortant
        self.strain = strain
        self.is_severe = is_severe
        return

    # compares two strains
    # if they both have the same antigenic segments we return homotypic
    def match(self, strainIn):
        numAgSegments = self.sim.numAgSegments
        if strainIn[:numAgSegments] == self.strain[:numAgSegments]:
            return PathogenMatch.HOMOTYPIC

        strains_match = False
        for i in range(numAgSegments):
            if strainIn[i] == self.strain[i]:
                strains_match = True

        if strains_match:
            return PathogenMatch.PARTIAL_HETERO
        else:
            return PathogenMatch.COMPLETE_HETERO

    def get_fitness(self):
        """ Get the fitness based on the fitness hypothesis and the two strains """
        key = (self.strain[0], self.strain[1])
        default = 0.90
        mapping = {
        		(1, 1): 0.93,
        		(2, 2): 0.93,
        		(3, 3): 0.93,
        		(4, 4): 0.93,
        	}
        return mapping.get(key, default)

    def get_strain_name(self):
        G,P,A,B = [str(self.strain[i]) for i in range(4)]
        return f'G{G}P{P}A{A}B{B}'

    def __str__(self):
        return "Strain: " + self.get_strain_name() + " Severe: " + str(self.is_severe) + " Host: " + str(self.host.id) + str(self.creation_time)


class Rota(ss.Infection):

    def __init__(self, **kwargs):
        """
        Create the simulation.

        Args:
            defaults (list): a list of parameters matching the command-line inputs; see below
            verbose (bool): the "verbosity" of the output: if False, print nothing; if None, print the timestep; if True, print out results
        """
        super().__init__()
        self.define_pars(
            reassortment_rate = 0.1,
            ve_i_to_ve_s_ratio = 0.5,
            rel_beta = 1.0,
            gamma = 365/7,
            omega = 365/50,
            contact_rate = 365/1,
            vaccination_time = 20,
            vaccine_efficacy_d1 = [0.6, 0.45, 0.15],
            vaccine_efficacy_d2 = [0.8, 0.65, 0.35],
            vaccination_single_dose_waning_rate = 365/273, #365/1273
            vaccination_double_dose_waning_rate = 365/546, #365/2600
        )
        self.update_pars(**kwargs)

        return



### RotaABM class
class Sim(ss.Sim):
    """
    Run the simulation
    """



    @staticmethod
    def get_probability_of_severe(sim, pathogen_in, vaccine, immunity_count): # TEMP: refactor and include above
        if immunity_count >= 3:
            severity_probability = 0.18
        elif immunity_count == 2:
            severity_probability = 0.24
        elif immunity_count == 1:
            severity_probability = 0.23
        elif immunity_count == 0:
            severity_probability = 0.17

        if vaccine is not None:
            # Probability of severity also depends on the strain (homotypic/heterltypic/etc.)
            pathogen_strain_type = pathogen_in.match(vaccine[0][0])
            # Effectiveness of the vaccination depends on the number of doses
            if vaccine[2] == 1:
                ve_s = sim.vaccine_efficacy_s_d1[pathogen_strain_type]
            elif vaccine[2] == 2:
                ve_s = sim.vaccine_efficacy_s_d2[pathogen_strain_type]
            else:
                raise NotImplementedError(f"Unsupported vaccine dose: {vaccine[2]}")
            return severity_probability * (1-ve_s)
        else:
            return severity_probability

    @staticmethod
    def coInfected_contacts(host1, host2, strain_counts):
        h2existing_pathogens = list(host2.infecting_pathogen)
        randomnumber = rnd.random()
        if randomnumber < 0.02:       # giving all the possible strains
            for path in host1.infecting_pathogen:
                if host2.can_variant_infect_host(path.strain, h2existing_pathogens):
                    host2.infect_with_pathogen(path, strain_counts)
        else:  # give only one strain depending on fitness
            host1paths = list(host1.infecting_pathogen)
            # Sort by fitness first and randomize the ones with the same fitness
            host1paths.sort(key=lambda path: (path.get_fitness(), rnd.random()), reverse=True)
            for path in host1paths:
                if host2.can_variant_infect_host(path.strain, h2existing_pathogens):
                    infected = host2.infect_with_pathogen(path, strain_counts)
                    if infected:
                        break

    def contact_event(self, contacts, infected_pop, strain_count):
        if len(infected_pop) == 0:
            print("[Warning] No infected hosts in a contact event. Skipping")
            return

        h1_inds = np.random.randint(len(infected_pop), size=contacts)
        h2_inds = np.random.randint(len(self.host_pop), size=contacts)
        rnd_nums = np.random.random(size=contacts)
        counter = 0

        # based on prior infections and current infections, the relative risk of subsequent infections
        infecting_probability_map = {
            0: 1,
            1: 0.61,
            2: 0.48,
            3: 0.33,
        }

        for h1_ind, h2_ind, rnd_num in zip(h1_inds, h2_inds, rnd_nums):
            h1 = infected_pop[h1_ind]
            h2 = self.host_pop[h2_ind]

            while h1 == h2:
                h2 = rnd.choice(self.host_pop)

            infecting_probability = infecting_probability_map.get(h2.prior_infections, 0)
            infecting_probability *= self.rel_beta # Scale by this calibration parameter

            # No infection occurs
            if rnd_num > infecting_probability:
                continue # CK: was "return", which was a bug!
            else:
                counter += 1
                h2_previously_infected = h2.isInfected()

                if len(h1.infecting_pathogen)==1:
                    if h2.can_variant_infect_host(h1.infecting_pathogen[0].strain, h2.infecting_pathogen):
                        h2.infect_with_pathogen(h1.infecting_pathogen[0], strain_count)
                    # else:
                    #     print('Unclear what should happen here')
                else:
                    self.coInfected_contacts(h1,h2,strain_count)

                # in this case h2 was not infected before but is infected now
                if not h2_previously_infected and h2.isInfected():
                    infected_pop.append(h2)
        return counter

    def get_weights_by_age(self):
        bdays = np.array(self.host_pop.bdays)
        weights = self.t - bdays
        total_w = np.sum(weights)
        weights = weights / total_w
        return weights

    def death_event(self, num_deaths, infected_pop, strain_count):
        host_list = np.arange(len(self.host_pop))
        p = self.get_weights_by_age()
        inds = np.random.choice(host_list, p=p, size=num_deaths, replace=False)
        dying_hosts = [self.host_pop[ind] for ind in inds]
        for h in dying_hosts:
            if h.isInfected():
                infected_pop.remove(h)
                for path in h.infecting_pathogen:
                    if not path.is_reassortant:
                        strain_count[path.strain] -= 1
            if h.is_immune_flag:
                self.immunity_counts -= 1
            self.host_pop.remove(h)
        return

    def recovery_event(self, num_recovered, infected_pop, strain_count):
        weights=np.array([x.get_oldest_current_infection() for x in infected_pop])
        # If there is no one with an infection older than 0 return without recovery
        if (sum(weights) == 0):
            return
        # weights_e = np.exp(weights)
        total_w = np.sum(weights)
        weights = weights / total_w

        recovering_hosts = np.random.choice(infected_pop, p=weights, size=num_recovered, replace=False)
        for host in recovering_hosts:
            if not host.is_immune_flag:
                self.immunity_counts +=1
            host.recover(strain_count)
            infected_pop.remove(host)

    @staticmethod
    def reassortment_event(infected_pop, reassortment_count):
        coinfectedhosts = []
        for i in infected_pop:
            if len(i.infecting_pathogen) >= 2:
                coinfectedhosts.append(i)
        rnd.shuffle(coinfectedhosts) # TODO: maybe replace this

        for i in range(min(len(coinfectedhosts),reassortment_count)):
            parentalstrains = [path.strain for path in coinfectedhosts[i].infecting_pathogen]
            possible_reassortants = [path for path in coinfectedhosts[i].compute_combinations() if path not in parentalstrains]
            for path in possible_reassortants:
                coinfectedhosts[i].infect_with_reassortant(path)

    def waning_event(self, wanings):
        # Get all the hosts in the population that has an immunity
        h_immune = [h for h in self.host_pop if h.is_immune_flag]
        oldest = np.array([h.oldest_infection for h in h_immune])
        # oldest += 1e-6*np.random.random(len(oldest)) # For tiebreaking -- not needed
        order = np.argsort(oldest)

        # For the selcted hosts set the immunity to be None
        for i in order[:wanings]:#range(min(len(hosts_with_immunity), wanings)):
            h = h_immune[i]
            h.immunity =  {}
            h.is_immune_flag = False
            h.oldest_infection = np.nan
            h.prior_infections = 0
            self.immunity_counts -= 1

    @staticmethod
    def waning_vaccinations_first_dose(single_dose_pop, wanings):
        """ Get all the hosts in the population that has an vaccine immunity """
        rnd.shuffle(single_dose_pop)
        # For the selcted hosts set the immunity to be None
        for i in range(min(len(single_dose_pop), wanings)):
            h = single_dose_pop[i]
            h.vaccinations =  None

    @staticmethod
    def waning_vaccinations_second_dose(second_dose_pop, wanings):
        rnd.shuffle(second_dose_pop)
        # For the selcted hosts set the immunity to be None
        for i in range(min(len(second_dose_pop), wanings)):
            h = second_dose_pop[i]
            h.vaccinations =  None

    def birth_events(self, birth_count):
        for _ in range(birth_count):
            self.pop_id += 1
            new_host = Host(self.pop_id, sim=self)
            new_host.bday = self.t
            self.host_pop.append(new_host)
            if self.vaccine_hypothesis !=0 and self.done_vaccinated:
                if rnd.random() < self.vaccine_first_dose_rate:
                    self.to_be_vaccinated_pop.append(new_host)

    @staticmethod
    def get_strain_antigenic_name(strain):
        return "G" + str(strain[0]) + "P" + str(strain[1])

    @staticmethod
    def solve_quadratic(a, b, c):
        discriminant = b**2 - 4*a*c
        if discriminant >= 0:
            root1 = (-b + discriminant**0.5) / (2*a)
            root2 = (-b - discriminant**0.5) / (2*a)
            return tuple(sorted([root1, root2]))
        else:
            return "No real roots"

    def breakdown_vaccine_efficacy(self, ve, x):
        (r1, r2) = self.solve_quadratic(x, -(1+x), ve)
        if self.verbose: print(r1, r2)
        if r1 >= 0 and r1 <= 1:
            ve_s = r1
        elif r2 >= 0 and r2 <= 1:
            ve_s = r2
        else:
            raise RuntimeError("No valid solution to the equation: x: %d, ve: %d. Solutions: %f %f" % (x, ve, r1, r2))
        ve_i = x * ve_s
        return (ve_i, ve_s)


    def prepare_run(self):
        """
        Set up the variables for the run
        """

        self.homotypic_immunity_rate = 0
        self.partial_cross_immunity_rate = 1
        self.complete_heterotypic_immunity_rate = 0

        self.vaccine_efficacy_i_d1 = {}
        self.vaccine_efficacy_s_d1 = {}
        self.vaccine_efficacy_i_d2 = {}
        self.vaccine_efficacy_s_d2 = {}
        for (k, v) in self.vaccine_efficacy_d1.items():
            (ve_i, ve_s) = self.breakdown_vaccine_efficacy(v, self.ve_i_to_ve_s_ratio)
            self.vaccine_efficacy_i_d1[k] = ve_i
            self.vaccine_efficacy_s_d1[k] = ve_s
        for (k, v) in self.vaccine_efficacy_d2.items():
            (ve_i, ve_s) = self.breakdown_vaccine_efficacy(v, self.ve_i_to_ve_s_ratio)
            self.vaccine_efficacy_i_d2[k] = ve_i
            self.vaccine_efficacy_s_d2[k] = ve_s

        # Vaccination rates are derived based on the following formula
        self.vaccine_second_dose_rate = 0.8
        self.vaccine_first_dose_rate = math.sqrt(self.vaccine_second_dose_rate)
        if self.verbose: print("Vaccination - first dose rate: %s, second dose rate %s" % (self.vaccine_first_dose_rate, self.vaccine_second_dose_rate))

        self.total_strain_counts_vaccine = {}

        numSegments = 4
        numNoneAgSegments = 2
        self.numAgSegments = numSegments - numNoneAgSegments
        segmentVariants = [[1,2,3,4,9,11,12], [8,4,6], [i for i in range(1, 2)], [i for i in range(1, 2)]]
        segment_combinations = [tuple(i) for i in itertools.product(*segmentVariants)]  # getting all possible combinations from a list of list
        rnd.shuffle(segment_combinations)
        number_all_strains = len(segment_combinations)
        n_init_seg = 100
        initial_segment_combinations = {
            (1,8,1,1) : n_init_seg,
            (2,4,1,1) : n_init_seg,
            (9,8,1,1) : n_init_seg,
            (4,8,1,1) : n_init_seg,
            (3,8,1,1) : n_init_seg,
            (12,8,1,1): n_init_seg,
            (12,6,1,1): n_init_seg,
            (9,4,1,1) : n_init_seg,
            (9,6,1,1) : n_init_seg,
            (1,6,1,1) : n_init_seg,
            (2,8,1,1) : n_init_seg,
            (2,6,1,1) : n_init_seg,
            (11,8,1,1): n_init_seg,
            (11,6,1,1): n_init_seg,
            (1,4,1,1) : n_init_seg,
            (12,4,1,1): n_init_seg,
        }
        # Track the number of immune hosts(immunity_counts) in the host population
        infected_pop = []
        pathogens_pop = []

        # for each strain track the number of hosts infected with it at current time: strain_count
        strain_count = {}

        # for each number in range of N, make a new Host object, i is the id.
        host_pop = HostPop(self.N, self)

        self.pop_id = self.N
        self.to_be_vaccinated_pop = []
        self.single_dose_vaccinated_pop = []

        # Store these for later
        self.infected_pop = infected_pop
        self.pathogens_pop = pathogens_pop
        self.host_pop = host_pop
        self.strain_count = strain_count

        for i in range(number_all_strains):
            self.strain_count[segment_combinations[i]] = 0

        # if initial immunity is true
        if self.verbose:
            if self.initial_immunity:
                print("Initial immunity is set to True")
            else:
                print("Initial immunity is set to False")

        ### infecting the initial infecteds
        for (initial_strain, num_infected) in initial_segment_combinations.items():
            if self.initial_immunity:
                for j in range(self.num_initial_immune):
                    h = rnd.choice(host_pop)
                    h.immunity[initial_strain] = self.t
                    self.immunity_counts += 1
                    h.is_immune_flag = True

            for j in range(num_infected):
                h = rnd.choice(host_pop)
                if not h.isInfected():
                    infected_pop.append(h)
                p = Pathogen(self, False, self.t, host = h, strain = initial_strain)
                pathogens_pop.append(p)
                h.infecting_pathogen.append(p)
                strain_count[p.strain] += 1
        if self.verbose: print(strain_count)

        for strain, count in strain_count.items():
            if strain[:self.numAgSegments] in self.total_strain_counts_vaccine:
                self.total_strain_counts_vaccine[strain[:self.numAgSegments]] += count
            else:
                self.total_strain_counts_vaccine[strain[:self.numAgSegments]] = count
        return

    def integrate(self):
        """
        Perform the actual integration loop
        """
        host_pop = self.host_pop
        strain_count = self.strain_count
        infected_pop = self.infected_pop
        single_dose_vaccinated_pop = self.single_dose_vaccinated_pop
        to_be_vaccinated_pop = self.to_be_vaccinated_pop
        total_strain_counts_vaccine = self.total_strain_counts_vaccine

        self.event_dict = sc.objdict(
            births=0,
            deaths=0,
            recoveries=0,
            contacts=0,
            wanings=0,
            reassortments=0,
            vaccine_dose_1_wanings=0,
            vaccine_dose_2_wanings=0,
        )

        self.T = sc.timer() # To track the time it takes to run the simulation
        while self.t<self.timelimit:
            if self.tau_steps % 10 == 0:
                if self.verbose is not False: print(f"Year: {self.t:n}; step: {self.tau_steps}; hosts: {len(host_pop)}; elapsed: {self.T.total:n} s")
                if self.verbose: print(self.strain_count)

            ### Every 100 steps, write the age distribution of the population to a file
            if self.tau_steps % 100 == 0:
                age_dict = {}
                for age_range in age_labels:
                    age_dict[age_range] = 0
                for h in host_pop:
                    age_dict[h.get_age_category()] += 1
                if self.verbose: print("Ages: ", age_dict)

            # Count the number of hosts with 1 or 2 vaccinations
            single_dose_hosts = []
            double_dose_hosts = []
            for h in host_pop:
                if h.vaccine is not None:
                    if h.vaccine[2] == 1:
                        single_dose_hosts.append(h)
                    elif h.vaccine[2] == 2:
                        double_dose_hosts.append(h)

            # Get the number of events in a single tau step
            events = self.get_event_counts(len(host_pop), len(infected_pop), self.immunity_counts, self.tau, self.reassortmentRate_GP, len(single_dose_hosts), len(double_dose_hosts))
            births, deaths, recoveries, contacts, wanings, reassortments, vaccine_dose_1_wanings, vaccine_dose_2_wanings = events

            # Parse into dict
            self.event_dict[:] += events

            # perform the events for the obtained counts
            self.reassortment_event(infected_pop, reassortments) # calling the function
            self.contact_event(contacts, infected_pop, strain_count)
            self.recovery_event(recoveries, infected_pop, strain_count)
            self.waning_event(wanings)
            self.waning_vaccinations_first_dose(single_dose_hosts, vaccine_dose_1_wanings)
            self.waning_vaccinations_second_dose(double_dose_hosts, vaccine_dose_2_wanings)

            # Collect the total counts of strains at each time step to determine the most prevalent strain for vaccination
            if not self.done_vaccinated:
                for strain, count in strain_count.items():
                    total_strain_counts_vaccine[strain[:self.numAgSegments]] += count

            # Administer the first dose of the vaccine
            # Vaccination strain is the most prevalent strain in the population before the vaccination starts
            if self.vaccine_hypothesis!=0 and (not self.done_vaccinated) and self.t >= self.vaccination_time:
                # Sort the strains by the number of hosts infected with it in the past
                # Pick the last one from the sorted list as the most prevalent strain
                vaccinated_strain = sorted(list(total_strain_counts_vaccine.keys()), key=lambda x: total_strain_counts_vaccine[x])[-1]
                # Select hosts under 6.5 weeks and over 4.55 weeks of age for vaccinate
                child_host_pop = [h for h in host_pop if self.t - h.bday <= 0.13 and self.t - h.bday >= 0.09]
                # Use the vaccination rate to determine the number of hosts to vaccinate
                vaccination_count = int(len(child_host_pop)*self.vaccine_first_dose_rate)
                sample_population = rnd.sample(child_host_pop, vaccination_count)
                for h in sample_population:
                    h.vaccinate(vaccinated_strain)
                    single_dose_vaccinated_pop.append(h)
                self.done_vaccinated = True
            elif self.done_vaccinated:
                for child in to_be_vaccinated_pop:
                    if self.t - child.bday >= 0.11:
                        child.vaccinate(vaccinated_strain)
                        to_be_vaccinated_pop.remove(child)
                        single_dose_vaccinated_pop.append(child)

            # Administer the second dose of the vaccine if first dose has already been administered.
            # The second dose is administered 6 weeks after the first dose with probability vaccine_second_dose_rate
            if self.done_vaccinated:
                while len(single_dose_vaccinated_pop) > 0:
                    # If the first dose of the vaccine is older than 6 weeks then administer the second dose
                    if self.t - single_dose_vaccinated_pop[0].vaccine[1] >= 0.11:
                        child = single_dose_vaccinated_pop.pop(0)
                        if rnd.random() < self.vaccine_second_dose_rate:
                            child.vaccinate(vaccinated_strain)
                    else:
                        break

            f = self.files
            if self.t >= self.last_data_colllected:
                self.collect_and_write_data(f.sample_outputfilename, f.vaccinations_outputfilename, f.sample_vaccine_efficacy_output_filename, sample=True)
                self.collect_and_write_data(f.infected_all_outputfilename, f.vaccinations_outputfilename, f.vaccine_efficacy_output_filename, sample=False)
                self.last_data_colllected += self.data_collection_rate

            self.tau_steps += 1
            self.t += self.tau

        if self.verbose is not False:
            self.T.toc()
            print(self.event_dict)
        return self.event_dict

    def run(self):
        """
        Run the simulation
        """
        self.prepare_run()
        events = self.integrate()
        self.to_df()
        return events


if __name__ == '__main__':
    sim = Sim(n_agents=10_000, timelimit=2)
    sim.run()

