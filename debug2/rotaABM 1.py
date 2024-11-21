#########################################
import csv
import itertools
from re import T
import numpy as np
import random as rnd
from datetime import datetime
import time
from enum import Enum
import sys
import math
import sciris as sc


def main(defaults=None, verbose=None):
    """
    The main script used to run the simulation.
    
    Args:
        defaults (list): a list of parameters matching the command-line inputs; see below
        verbose (bool): the "verbosity" of the output: if False, print nothing; if None, print the timestep; if True, print out results
    """

    global immunityCounts  
    global pop_id
    global t
    
    args = sys.argv
    if defaults is None:
        defaults = ['', # Placeholder (file name)
            1,   # immunity_hypothesis. Defines the immunity rates for Homotypic, Partial Heterotypic, Complete Heterotypic infections.
            0.1, # reassortment_rate
            1,   # fitness_hypothesis. Defines how the fitness is computed for a strain.
            1,   # vaccine_hypothesis
            1,   # waning_hypothesis
            0,   # initial_immunity
            0.5, # ve_i_to_ve_s_ratio
            1,   # experimentNumber
        ]
    if verbose is not False: print(args)
    if len(args) < 8:
        args = args + defaults[len(args):]
    
    immunity_hypothesis = int(args[1])
    reassortment_rate = float(args[2])
    fitness_hypothesis = int(args[3])
    vaccine_hypothesis = int(args[4])
    waning_hypothesis = int(args[5])
    initial_immunity = int(args[6]) # 0 = no immunity
    ve_i_to_ve_s_ratio = float(args[7])
    experimentNumber = int(args[8])
    
    now = datetime.now() # current date and time
    date_time = now.strftime("%m_%d_%Y_%H_%M")
    if verbose is not False: print("date and time:", date_time)

    myseed = experimentNumber
    rnd.seed(myseed)
    np.random.seed(myseed)
    
    name_suffix =  '%r_%r_%r_%r_%r_%r_%r_%r' % (immunity_hypothesis, reassortment_rate, fitness_hypothesis, vaccine_hypothesis, waning_hypothesis, initial_immunity, ve_i_to_ve_s_ratio, experimentNumber)

    outputfilename = './results/rota_straincount_%s.csv' % (name_suffix)
    vaccinations_outputfilename = './results/rota_vaccinecount_%s.csv' % (name_suffix)
    sample_outputfilename = './results/rota_strains_sampled_%s.csv' % (name_suffix)
    infected_all_outputfilename = './results/rota_strains_infected_all_%s.csv' % (name_suffix)
    age_outputfilename = './results/rota_agecount_%s.csv' % (name_suffix)
    vaccine_efficacy_output_filename = './results/rota_vaccine_efficacy_%s.csv' % (name_suffix)
    sample_vaccine_efficacy_output_filename = './results/rota_sample_vaccine_efficacy_%s.csv' % (name_suffix)
    
    # Initialize all the output files
    def initialize_files(strainCount):
        with open(outputfilename, "w+", newline='') as outputfile:
            write = csv.writer(outputfile)
            write.writerow(["time"] + list(strainCount.keys()) + ["ReassortmentCount"])  # header for the csv file
            write.writerow([t] + list(strainCount.values()) + [ReassortmentCount])  # first row of the csv file will be the initial state
    
        with open(sample_outputfilename, "w+", newline='') as outputfile:
            write = csv.writer(outputfile)
            write.writerow(["id", "Strain", "CollectionTime", "Age", "Severity", "InfectionTime", "PopulationSize"])
        with open(infected_all_outputfilename, "w+", newline='') as outputfile:
            write = csv.writer(outputfile)
            write.writerow(["id", "Strain", "CollectionTime", "Age", "Severity", "InfectionTime", "PopulationSize"])
        with open(vaccinations_outputfilename, "w+", newline='') as outputfile:
            write = csv.writer(outputfile)
            write.writerow(["id", "VaccineStrain", "CollectionTime", "Age", "Dose"])  # header for the csv file        
    
        for outfile in [vaccine_efficacy_output_filename, sample_vaccine_efficacy_output_filename]:
            with open(outfile, "w+", newline='') as outputfile:
                write = csv.writer(outputfile)
                write.writerow(["CollectionTime", "Vaccinated", "Unvaccinated", "VaccinatedInfected", "VaccinatedSevere", "UnVaccinatedInfected", "UnVaccinatedSevere", 
                                "VaccinatedHomotypic", "VaccinatedHomotypicSevere", "VaccinatedpartialHetero", "VaccinatedpartialHeteroSevere", "VaccinatedFullHetero", "VaccinatedFullHeteroSevere"])
    
        with open(age_outputfilename, "w+", newline='') as outputfile:
            write = csv.writer(outputfile)
            write.writerow(["time"] + list(host.age_labels))    
    
    ############## Class Host ###########################
    class host(object): ## host class
        # Define age bins and labels
        age_bins = [2/12, 4/12, 6/12, 12/12, 24/12, 36/12, 48/12, 60/12, 100]
        age_distribution = [0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.84]
        age_labels = ['0-2', '2-4', '4-6', '6-12', '12-24', '24-36', '36-48', '48-60', '60+']
    
        def __init__(self, id):
            self.id = id
            self.bday = t - host.get_random_age()
            # set of strains the host is immune to
            self.immunity = {}
            self.vaccine = None
            self.infecting_pathogen = []
            self.priorInfections = 0
            self.prior_vaccinations = []
            self.infections_with_vaccination = []
            self.infections_without_vaccination = []
    
        def get_random_age():
            # pick a age bin
            random_age_bin = np.random.choice(list(range(len(host.age_bins))), p=host.age_distribution)
            # generate a random age in the bin
            if random_age_bin > 0:
                min_age = host.age_bins[random_age_bin-1]
            else:
                min_age = 0
            max_age = host.age_bins[random_age_bin]
            return rnd.uniform(min_age, max_age)
    
        def get_age_category(self):
            # Bin the age into categories
            for i in range(len(host.age_bins)):
                if t - self.bday < host.age_bins[i]:
                    return self.age_labels[i]
            return self.age_labels[-1]
        
        def get_oldest_current_infection(self):
            max_infection_times = max([t - p.creation_time for p in self.infecting_pathogen])
            return max_infection_times
    
        def get_oldest_infection(self):
            max_infection_times = max([t - p[1] for p in self.immunity.items()])
            return max_infection_times
        
        def computePossibleCombinations(self):
            segCombinations = []
    
            # We want to only reassort the GP types
            # Assumes that antigenic segments are at the start
            for i in range(numAgSegments):
                availableVariants = set([])                 
                for j in self.infecting_pathogen:
                    availableVariants.add((j.strain[i]))
                segCombinations.append(availableVariants)
    
            # compute the parental strains (duplicate work?)
            parantal_strains = [j.strain[:numAgSegments] for j in self.infecting_pathogen]
    
            # Itertools product returns all possible combinations
            # We are only interested in strain combinations that are reassortants of the parental strains
            # We need to skip all existing combinations from the parents
            # Ex: (1, 1, 2, 2) and (2, 2, 1, 1) should not create (1, 1, 1, 1) as a possible reassortant if only the antigenic parts reassort
            all_antigenic_combinations = [i for i in itertools.product(*segCombinations) if i not in parantal_strains]
            all_nonantigenic_combinations = [j.strain[numAgSegments:] for j in self.infecting_pathogen]
            all_strains = set([(i[0] + i[1]) for i in itertools.product(all_antigenic_combinations, all_nonantigenic_combinations)])
            
    
            #print("Parents: ", [j.strain for j in self.infecting_pathogen], "Reassortants: ", all_strains)
            all_pathogens = [pathogen(True, t, host = self, strain=tuple(i)) for i in all_strains]
    
            # The commented code below is for the old version where all parts reassort
            return all_pathogens
    
        def getPossibleCombinations(self):
            return self.computePossibleCombinations()
    
        def isInfected(self):
            return len(self.infecting_pathogen) != 0
    
        def recover(self,strainCounts):
            # We will use the pathogen creation time to count the number of infections
            creation_times = set()
            for path in self.infecting_pathogen:
                if not path.is_reassortant:
                    strainCounts[path.strain] -= 1
                    creation_times.add(path.creation_time)
                    self.immunity[path.strain] = t
            self.priorInfections += len(creation_times)
            self.infecting_pathogen = []                  
            self.possibleCombinations = []
        
        def isImmune(self):
            return len(self.immunity) != 0
        
        def vaccinate(self, vaccinated_strain):
            if len(self.prior_vaccinations) == 0:
                self.prior_vaccinations.append(vaccinated_strain)
                self.vaccine = ([vaccinated_strain], t, 1)
            else:
                self.prior_vaccinations.append(vaccinated_strain)
                self.vaccine = ([vaccinated_strain], t, 2)
        
        def isVaccineimmune(self, infecting_strain):
            # Effectiveness of the vaccination depends on the number of doses
            if self.vaccine[2] == 1:
                ve_i_rates = vaccine_efficacy_i_d1
            elif self.vaccine[2] == 2:
                ve_i_rates = vaccine_efficacy_i_d2
            else:
                print("Unsupported vaccine dose")
                exit(-1)

            # Vaccine strain only contains the antigenic parts
            vaccine_strain = self.vaccine[0]
            
            if vaccine_hypothesis == 0:
                return False
            if vaccine_hypothesis == 1:            
                if infecting_strain[:numAgSegments] in vaccine_strain:
                    if rnd.random() < ve_i_rates[PathogenMatch.HOMOTYPIC]:
                        return True
                    else:
                        return False
            elif vaccine_hypothesis == 2:
                if infecting_strain[:numAgSegments] in vaccine_strain:
                    if rnd.random() < ve_i_rates[PathogenMatch.HOMOTYPIC]:
                        return True
                    else:
                        return False
                strains_match = False
                for i in range(numAgSegments):         
                    immune_genotypes = [strain[i] for strain in vaccine_strain]
                    if infecting_strain[i] in immune_genotypes:
                        strains_match = True
                if strains_match:
                    if rnd.random() < ve_i_rates[PathogenMatch.PARTIAL_HETERO]:
                        return True
                else:
                    return False
            elif vaccine_hypothesis == 3:
                if infecting_strain[:numAgSegments] in vaccine_strain:
                    if rnd.random() < ve_i_rates[PathogenMatch.HOMOTYPIC]:
                        return True
                    else:
                        return False
                strains_match = False
                for i in range(numAgSegments):         
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
                print("Unsupported vaccine hypothesis")
                exit(-1)
        

        def can_variant_infect_host(self, infecting_strain, currentInfections):
            if (self.vaccine is not None) and self.isVaccineimmune(infecting_strain):
                return False
            
            if immunity_hypothesis == 1:
                current_infecting_strains = [i.strain[:numAgSegments] for i in currentInfections]
                if infecting_strain[:numAgSegments] in current_infecting_strains:
                    return False
                # Only immune if antigenic segments match exactly
                immune_strains = [s[:numAgSegments] for s in self.immunity.keys()]
                if infecting_strain[:numAgSegments] in immune_strains:
                    return False
                return True
            elif immunity_hypothesis == 2:
                current_infecting_strains = [i.strain[:numAgSegments] for i in currentInfections]
                if infecting_strain[:numAgSegments] in current_infecting_strains:
                    return False
                # Completely immune for partial heterotypic strains
                for i in range(numAgSegments):         
                    immune_genotypes = [strain[i] for strain in self.immunity.keys()]
                    if infecting_strain[i] in immune_genotypes:
                        return False
                return True
            elif immunity_hypothesis == 3:
                current_infecting_strains = [i.strain[:numAgSegments] for i in currentInfections]
                if infecting_strain[:numAgSegments] in current_infecting_strains:
                    return False
                
                # completely immune if antigenic segments match exactly
                immune_strains = [s[:numAgSegments] for s in self.immunity.keys()]
                if infecting_strain[:numAgSegments] in immune_strains:
                    return False
    
                # Partial heterotypic immunity if not
                shared_genotype = False
                for i in range(numAgSegments):         
                    immune_genotypes = [strain[i] for strain in self.immunity.keys()]
                    if infecting_strain[i] in immune_genotypes:
                        shared_genotype = True
                if shared_genotype:
                    temp = rnd.random()
                    if temp<partialCrossImmunityRate:
                        return False
                return True
            elif immunity_hypothesis == 4:
                current_infecting_strains = [i.strain[:numAgSegments] for i in currentInfections]
                if infecting_strain[:numAgSegments] in current_infecting_strains:
                    return False
                
                # completely immune if antigenic segments match exactly
                immune_strains = [s[:numAgSegments] for s in self.immunity.keys()]
                if infecting_strain[:numAgSegments] in immune_strains:
                    return False

                # Partial heterotypic immunity if not
                shared_genotype = False
                for i in range(numAgSegments):         
                    immune_genotypes = [strain[i] for strain in self.immunity.keys()]
                    if infecting_strain[i] in immune_genotypes:
                        shared_genotype = True
                if shared_genotype:
                    temp = rnd.random()
                    if temp<partialCrossImmunityRate:
                        return False
                else:
                    temp = rnd.random()
                    if temp<completeHeterotypicImmunityrate:
                        return False
                return True
            elif immunity_hypothesis == 5:
                current_infecting_strains = [i.strain[:numAgSegments] for i in currentInfections]
                if infecting_strain[:numAgSegments] in current_infecting_strains:
                    return False
                
                # Partial heterotypic immunity
                shared_genotype = False      
                immune_ptypes = [strain[1] for strain in self.immunity.keys()]
                if infecting_strain[1] in immune_ptypes:
                    return False
                else:
                    return True
            elif immunity_hypothesis == 6:
                current_infecting_strains = [i.strain[:numAgSegments] for i in currentInfections]
                if infecting_strain[:numAgSegments] in current_infecting_strains:
                    return False
                # Partial heterotypic immunity
                shared_genotype = False      
                immune_gtypes = [strain[0] for strain in self.immunity.keys()]
                if infecting_strain[0] in immune_ptypes:
                    return False
                else:
                    return True
            elif immunity_hypothesis == 7 or immunity_hypothesis == 8 or immunity_hypothesis == 9 or immunity_hypothesis == 10:
                current_infecting_strains = [i.strain[:numAgSegments] for i in currentInfections]
                if infecting_strain[:numAgSegments] in current_infecting_strains:
                    return False
                
                # completely immune if antigenic segments match exactly
                immune_strains = [s[:numAgSegments] for s in self.immunity.keys()]
                if infecting_strain[:numAgSegments] in immune_strains:
                    temp = rnd.random()
                    if temp<HomotypicImmunityRate:
                        return False

                # Partial heterotypic immunity if not
                shared_genotype = False
                for i in range(numAgSegments):         
                    immune_genotypes = [strain[i] for strain in self.immunity.keys()]
                    if infecting_strain[i] in immune_genotypes:
                        shared_genotype = True
                if shared_genotype:
                    temp = rnd.random()
                    if temp<partialCrossImmunityRate:
                        return False
                else:
                    temp = rnd.random()
                    if temp<completeHeterotypicImmunityrate:
                        return False
                return True
            else:
                print("[Error] Immunity hypothesis not implemented")
                exit(-1)
    
        def record_infection(self, new_p):        
            if len(self.prior_vaccinations) != 0:
                vaccine_strain = self.prior_vaccinations[-1]            
                self.infections_with_vaccination.append((new_p, new_p.match(vaccine_strain)))
            else:
                self.infections_without_vaccination.append(new_p)
                
        def infect_with_pathogen(self, pathogenIn, strainCounts):
            #this function returns a fitness value to a strain based on the hypo. 
            fitness = pathogenIn.getFitness()
            
            # e.g. fitness = 0.8 (theres a 80% chance the virus infecteing a host).then 20% of the time virus will not infect the host. 
            # in the following if condition, we generate random number between 0 and 1. 
            # this random number has a 20% chance of being greater than 0.8.
            # in this case we go inside the if condition, and return back without going forward with the remaining function.
            # if it is false i.e random number is less than 0.8, we infect the host. 
            if rnd.random()> fitness:                
                return False
            
            # Probability of getting a severe decease depends on the vaccination status of the host and the number of previous infections
            severity_probability = get_probability_of_severe(pathogenIn, self.vaccine, self.priorInfections)
            if rnd.random() < severity_probability:
                severe = True
            else:
                severe = False
    
            new_p = pathogen(False, t, host = self, strain= pathogenIn.strain, is_severe=severe)
            self.infecting_pathogen.append(new_p)
            self.record_infection(new_p)
    
            strainCounts[new_p.strain] += 1
    
            return True
    
        def infect_with_reassortant(self, reassortant_virus):
            self.infecting_pathogen.append(reassortant_virus)
    
    class PathogenMatch(Enum):
        COMPLETE_HETERO = 1
        PARTIAL_HETERO = 2
        HOMOTYPIC = 3
    
    ############## class Pathogen ###########################
    class pathogen(object): 
        def __init__(self, is_reassortant, creation_time, is_severe=False, host=None, strain=None): 
            self.host = host
            self.creation_time = creation_time
            self.is_reassortant = is_reassortant
            self.strain = strain
            self.is_severe = is_severe
    
        def death(self):
            pathogens_pop.remove(self)
    
        def match(self, strainIn):
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
                
        def getFitness(self):
            if fitness_hypothesis == 1:
                return 1
            elif fitness_hypothesis == 2:
                if self.strain[0] == 1 and self.strain[1] == 1:
                    return 0.93
                elif self.strain[0] == 2 and self.strain[1] == 2:
                    return 0.93
                elif self.strain[0] == 3 and self.strain[1] == 3:
                    return 0.93
                elif self.strain[0] == 4 and self.strain[1] == 4:
                    return 0.93
                else:
                    return 0.90
            
            elif fitness_hypothesis == 3:
                if self.strain[0] == 1 and self.strain[1] == 1:
                    return 0.93
                elif self.strain[0] == 2 and self.strain[1] == 2:
                    return 0.93
                elif self.strain[0] == 3 and self.strain[1] == 3:
                    return 0.90
                elif self.strain[0] == 4 and self.strain[1] == 4:
                    return 0.90
                else:
                    return 0.87
                
            elif fitness_hypothesis == 4:
                if self.strain[0] == 1 and self.strain[1] == 1:
                    return 1
                elif self.strain[0] == 2 and self.strain[1] == 2:
                    return 0.2
                else:
                    return 1
            
            elif fitness_hypothesis == 5:
                if self.strain[0] == 1 and self.strain[1] == 1:
                    return 1
                elif self.strain[0] == 2 and self.strain[1] == 1 or self.strain[0] == 1 and self.strain[1] == 3:
                    return 0.5
                else:
                    return 0.2 
            elif fitness_hypothesis == 6:
                if self.strain[0] == 1 and self.strain[1] == 8:
                    return 1
                elif self.strain[0] == 2 and self.strain[1] == 4:
                    return 0.2
                elif self.strain[0] == 3 and self.strain[1] == 8:
                    return 0.4
                elif self.strain[0] == 4 and self.strain[1] == 8:
                    return 0.5
                else:
                    return 0.05
            elif fitness_hypothesis == 7:
                if self.strain[0] == 1 and self.strain[1] == 8:
                    return 1
                elif self.strain[0] == 2 and self.strain[1] == 4:
                    return 0.3
                elif self.strain[0] == 3 and self.strain[1] == 8:
                    return 0.7
                elif self.strain[0] == 4 and self.strain[1] == 8:
                    return 0.6
                else:
                    return 0.05
            elif fitness_hypothesis == 8:
                if self.strain[0] == 1 and self.strain[1] == 8:
                    return 1
                elif self.strain[0] == 2 and self.strain[1] == 4:
                    return 0.4
                elif self.strain[0] == 3 and self.strain[1] == 8:
                    return 0.9
                elif self.strain[0] == 4 and self.strain[1] == 8:
                    return 0.8
                else:
                    return 0.05
            elif fitness_hypothesis == 9:
                if self.strain[0] == 1 and self.strain[1] == 8:
                    return 1
                elif self.strain[0] == 2 and self.strain[1] == 4:
                    return 0.5
                elif self.strain[0] == 3 and self.strain[1] == 8:
                    return 0.9
                elif self.strain[0] == 4 and self.strain[1] == 8:
                    return 0.8
                else:
                    return 0.2
            elif fitness_hypothesis == 10:
                if self.strain[0] == 1 and self.strain[1] == 8:
                    return 1
                elif self.strain[0] == 2 and self.strain[1] == 4:
                    return 0.6
                elif self.strain[0] == 3 and self.strain[1] == 8:
                    return 0.9
                elif self.strain[0] == 4 and self.strain[1] == 8:
                    return 0.9
                else:
                    return 0.4
            elif fitness_hypothesis == 11:
                if self.strain[0] == 1 and self.strain[1] == 8:
                    return 0.98
                elif self.strain[0] == 2 and self.strain[1] == 4:
                    return 0.7
                elif self.strain[0] == 3 and self.strain[1] == 8:
                    return 0.8
                elif self.strain[0] == 4 and self.strain[1] == 8:
                    return 0.8
                else:
                    return 0.5
            elif fitness_hypothesis == 12:
                if self.strain[0] == 1 and self.strain[1] == 8:
                    return 0.98
                elif self.strain[0] == 2 and self.strain[1] == 4:
                    return 0.8
                elif self.strain[0] == 3 and self.strain[1] == 8:
                    return 0.9
                elif self.strain[0] == 4 and self.strain[1] == 8:
                    return 0.9
                else:
                    return 0.5
            elif fitness_hypothesis == 13:
                if self.strain[0] == 1 and self.strain[1] == 8:
                    return 0.98
                elif self.strain[0] == 2 and self.strain[1] == 4:
                    return 0.8
                elif self.strain[0] == 3 and self.strain[1] == 8:
                    return 0.9
                elif self.strain[0] == 4 and self.strain[1] == 8:
                    return 0.9
                else:
                    return 0.7
            elif fitness_hypothesis == 14:
                if self.strain[0] == 1 and self.strain[1] == 8:
                    return 0.98
                elif self.strain[0] == 2 and self.strain[1] == 4:
                    return 0.4
                elif self.strain[0] == 3 and self.strain[1] == 8:
                    return 0.7
                elif self.strain[0] == 4 and self.strain[1] == 8:
                    return 0.6
                elif self.strain[0] == 9 and self.strain[1] == 8:
                    return 0.7
                elif self.strain[0] == 12 and self.strain[1] == 8:
                    return 0.75
                elif self.strain[0] == 9 and self.strain[1] == 6:
                    return 0.58
                elif self.strain[0] == 11 and self.strain[1] == 8:
                    return 0.2
                else:
                    return 0.05
            elif fitness_hypothesis == 15:
                if self.strain[0] == 1 and self.strain[1] == 8:
                    return 1
                elif self.strain[0] == 2 and self.strain[1] == 4:
                    return 0.7
                elif self.strain[0] == 3 and self.strain[1] == 8:
                    return 0.93
                elif self.strain[0] == 4 and self.strain[1] == 8:
                    return 0.93
                elif self.strain[0] == 9 and self.strain[1] == 8:
                    return 0.95
                elif self.strain[0] == 12 and self.strain[1] == 8:
                    return 0.94
                elif self.strain[0] == 9 and self.strain[1] == 6:
                    return 0.3
                elif self.strain[0] == 11 and self.strain[1] == 8:
                    return 0.35
                else:
                    return 0.4
            elif fitness_hypothesis == 16:
                if self.strain[0] == 1 and self.strain[1] == 8:
                    return 1
                elif self.strain[0] == 2 and self.strain[1] == 4:
                    return 0.7
                elif self.strain[0] == 3 and self.strain[1] == 8:
                    return 0.85
                elif self.strain[0] == 4 and self.strain[1] == 8:
                    return 0.88
                elif self.strain[0] == 9 and self.strain[1] == 8:
                    return 0.95
                elif self.strain[0] == 12 and self.strain[1] == 8:
                    return 0.93
                elif self.strain[0] == 9 and self.strain[1] == 6:
                    return 0.85
                elif self.strain[0] == 12 and self.strain[1] == 6:
                    return 0.90
                elif self.strain[0] == 9 and self.strain[1] == 4:
                    return 0.90
                elif self.strain[0] == 1 and self.strain[1] == 6:
                    return 0.6
                elif self.strain[0] == 2 and self.strain[1] == 8:
                    return 0.6
                elif self.strain[0] == 2 and self.strain[1] == 6:
                    return 0.6
                else:
                    return 0.4
            elif fitness_hypothesis == 17:
                if self.strain[0] == 1 and self.strain[1] == 8:
                    return 1
                elif self.strain[0] == 2 and self.strain[1] == 4:
                    return 0.85
                elif self.strain[0] == 3 and self.strain[1] == 8:
                    return 0.85
                elif self.strain[0] == 4 and self.strain[1] == 8:
                    return 0.88
                elif self.strain[0] == 9 and self.strain[1] == 8:
                    return 0.95
                elif self.strain[0] == 12 and self.strain[1] == 8:
                    return 0.93
                elif self.strain[0] == 9 and self.strain[1] == 6:
                    return 0.83
                elif self.strain[0] == 12 and self.strain[1] == 6:
                    return 0.90
                elif self.strain[0] == 9 and self.strain[1] == 4:
                    return 0.90
                elif self.strain[0] == 1 and self.strain[1] == 6:
                    return 0.8
                elif self.strain[0] == 2 and self.strain[1] == 8:
                    return 0.8
                elif self.strain[0] == 2 and self.strain[1] == 6:
                    return 0.8
                else:
                    return 0.7
            elif fitness_hypothesis == 18:
                if self.strain[0] == 1 and self.strain[1] == 8:
                    return 1
                elif self.strain[0] == 2 and self.strain[1] == 4:
                    return 0.92
                elif self.strain[0] == 3 and self.strain[1] == 8:
                    return 0.79
                elif self.strain[0] == 4 and self.strain[1] == 8:
                    return 0.81
                elif self.strain[0] == 9 and self.strain[1] == 8:
                    return 0.95
                elif self.strain[0] == 12 and self.strain[1] == 8:
                    return 0.89
                elif self.strain[0] == 9 and self.strain[1] == 6:
                    return 0.80
                elif self.strain[0] == 12 and self.strain[1] == 6:
                    return 0.86
                elif self.strain[0] == 9 and self.strain[1] == 4:
                    return 0.83
                elif self.strain[0] == 1 and self.strain[1] == 6:
                    return 0.75
                elif self.strain[0] == 2 and self.strain[1] == 8:
                    return 0.75
                elif self.strain[0] == 2 and self.strain[1] == 6:
                    return 0.75
                else:
                    return 0.65
            else:
                print("Invalid fitness_hypothesis: ", fitness_hypothesis)
                exit(-1)
            
        def get_strain_name(self):
            return "G" + str(self.strain[0]) + "P" + str(self.strain[1]) + "A" + str(self.strain[2]) + "B" + str(self.strain[3])
        
        def __str__(self): 
            return "Strain: " + self.get_strain_name() + " Severe: " + str(self.is_severe) + " Host: " + str(self.host.id) + str(self.creation_time)

    ############# tau-Function to calculate event counts ############################
    def get_event_counts(N, I, R, tau, RR_GP, single_dose_count, double_dose_count): 
        births = np.random.poisson(size=1, lam=tau*N*birth_rate)[0]
        deaths = np.random.poisson(size=1, lam=tau*N*mu)[0]
        recoveries = np.random.poisson(size=1, lam=tau*gamma*I)[0]
        contacts = np.random.poisson(size=1, lam=tau*cont*I)[0]
        wanings = np.random.poisson(size=1, lam=tau*omega*R)[0]
        reassortments = np.random.poisson(size=1, lam=tau*RR_GP*I)[0]
        vaccination_wanings_one_dose = np.random.poisson(size=1, lam=tau*vacinnation_single_dose_waning_rate*single_dose_count)[0]
        vaccination_wanings_two_dose = np.random.poisson(size=1, lam=tau*vacinnation_double_dose_waning_rate*double_dose_count)[0]
        return (births, deaths, recoveries, contacts, wanings, reassortments, vaccination_wanings_one_dose, vaccination_wanings_two_dose)
    
    def coInfected_contacts(host1, host2, strainCounts):  
        global ReassortmentCount
    
        h2existing_pathogens = list(host2.infecting_pathogen)
        randomnumber = rnd.random()
        if randomnumber < 0.02:       # giving all the possible strains
            for path in host1.infecting_pathogen:
                if host2.can_variant_infect_host(path.strain, h2existing_pathogens):
                    host2.infect_with_pathogen(path, strainCounts)
        else:  # give only one strain depending on fitness
            host1paths = list(host1.infecting_pathogen)     
            # Sort by fitness first and randomize the ones with the same fitness
            host1paths.sort(key=lambda path: (path.getFitness(), rnd.random()), reverse=True)
            for path in host1paths:
                if host2.can_variant_infect_host(path.strain, h2existing_pathogens):
                    infected = host2.infect_with_pathogen(path, strainCounts)
                    if infected:
                        break
                
    def contact_event(infected_pop, host_pop, strainCount):
        if len(infected_pop) == 0:
            print("[Warning] No infected hosts in a contact event. Skipping")
            return
        
        h1 = rnd.choice(infected_pop)
        h2 = rnd.choice(host_pop)
    
        while h1 == h2:
            h2 = rnd.choice(host_pop)

        # based on proir infections and current infections, the relative risk of subsequent infections
        """ number_of_current_infections = len(h2.infecting_pathogen) """
        number_of_current_infections = 0

        if h2.priorInfections + number_of_current_infections == 0:
            infecting_probability = 1
        elif h2.priorInfections + number_of_current_infections == 1:
            infecting_probability = 0.61
        elif h2.priorInfections + number_of_current_infections == 2:  
            infecting_probability = 0.48
        elif h2.priorInfections + number_of_current_infections == 3:
            infecting_probability = 0.33
        else:
            infecting_probability = 0         
        rnd_num = rnd.random()
        if rnd_num > infecting_probability:
            return 
    
        h2_previously_infected = h2.isInfected()
    
        if len(h1.infecting_pathogen)==1:
            if h2.can_variant_infect_host(h1.infecting_pathogen[0].strain, h2.infecting_pathogen):
                h2.infect_with_pathogen(h1.infecting_pathogen[0], strainCount)
        else:
            coInfected_contacts(h1,h2,strainCount)
        
        # in this case h2 was not infected before but is infected now
        if not h2_previously_infected and h2.isInfected():
            infected_pop.append(h2)
    
    def get_weights_by_age(host_pop):
        weights = np.array([t - x.bday for x in host_pop])
        total_w = np.sum(weights)
        weights = weights / total_w
        return weights
    
    def death_event(num_deaths, infected_pop, host_pop, strainCount):
        global immunityCounts
        host_list = np.arange(len(host_pop))
        p = get_weights_by_age(host_pop)
        inds = np.random.choice(host_list, p=p, size=num_deaths, replace=False)
        dying_hosts = [host_pop[ind] for ind in inds]
        for h in dying_hosts:
            if h.isInfected():
                infected_pop.remove(h)
                for path in h.infecting_pathogen:
                    if not path.is_reassortant:
                        strainCount[path.strain] -= 1
            if h.isImmune():
                immunityCounts -= 1
            host_pop.remove(h)
    
    def recovery_event(num_recovered, infected_pop, strainCount):
        global immunityCounts
    
        weights=np.array([x.get_oldest_current_infection() for x in infected_pop])
        # If there is no one with an infection older than 0 return without recovery
        if (sum(weights) == 0):
            return
        # weights_e = np.exp(weights)
        total_w = np.sum(weights)
        weights = weights / total_w
    
        recovering_hosts = np.random.choice(infected_pop, p=weights, size=num_recovered, replace=False)
        for host in recovering_hosts:
            if not host.isImmune():
                immunityCounts +=1 
            host.recover(strainCount)
            infected_pop.remove(host)
    
    def reassortment_event(infected_pop, reassortment_count):
        coinfectedhosts = []
        for i in infected_pop:
            if len(i.infecting_pathogen) >= 2:
                coinfectedhosts.append(i)
        rnd.shuffle(coinfectedhosts)
    
        for i in range(min(len(coinfectedhosts),reassortment_count)):
            parentalstrains = [path.strain for path in coinfectedhosts[i].infecting_pathogen]
            possible_reassortants = [path for path in coinfectedhosts[i].getPossibleCombinations() if path not in parentalstrains]
            for path in possible_reassortants:
                coinfectedhosts[i].infect_with_reassortant(path)
    
    def waning_event(host_pop, wanings):
        global immunityCounts
    
        # Get all the hosts in the population that has an immunity
        h_immune = [h for h in host_pop if h.isImmune()]
        age_tiebreak = lambda x: (x.get_oldest_infection(), rnd.random())
        hosts_with_immunity = sorted(h_immune, key=age_tiebreak, reverse=True)
        
        # Alternate implementation -- not faster, but left in as a placeholder 
        # immune_inds = sc.findinds([h.is_immune for h in host_pop])
        # ages = np.array([host_pop[i].get_oldest_infection() for i in immune_inds])
        # ages += np.random.rand(len(ages))*1e-12 # Add noise to break ties
        # immunity_sort_inds = np.argsort(ages)[::-1]
        # immunity_sort_inds = immunity_sort_inds[:wanings]
    
        # For the selcted hosts set the immunity to be None
        for i in range(min(len(hosts_with_immunity), wanings)):
            h = hosts_with_immunity[i]
            h.immunity =  {}
            h.is_immune = False
            h.priorInfections = 0
            immunityCounts -= 1
    
    def waning_vaccinations_first_dose(single_dose_pop, wanings):
        """ Get all the hosts in the population that has an vaccine immunity """
        rnd.shuffle(single_dose_pop)
        # For the selcted hosts set the immunity to be None
        for i in range(min(len(single_dose_pop), wanings)):
            h = single_dose_pop[i]
            h.vaccinations =  None
    
    def waning_vaccinations_second_dose(second_dose_pop, wanings):
        rnd.shuffle(second_dose_pop)
        # For the selcted hosts set the immunity to be None
        for i in range(min(len(second_dose_pop), wanings)):
            h = second_dose_pop[i]
            h.vaccinations =  None
    
    def birth_events(birth_count, host_pop):
        global pop_id
        global t
    
        for _ in range(birth_count):
            pop_id += 1
            new_host = host(pop_id)
            new_host.bday = t
            host_pop.append(new_host)
            if vaccine_hypothesis !=0 and done_vaccinated:
                if rnd.random() < vaccine_first_dose_rate:
                    to_be_vaccinated_pop.append(new_host)
    
    
    def get_strain_antigenic_name(strain):
        return "G" + str(strain[0]) + "P" + str(strain[1])
    
    def collect_and_write_data(host_population, output_filename, vaccine_output_filename, vaccine_efficacy_output_filename, sample=False, sample_size=1000):
        """
        Collects data from the host population and writes it to a CSV file.
        If sample is True, it collects data from a random sample of the population.
        
        Args:
        - host_population: List of host objects.
        - output_filename: Name of the file to write the data.
        - sample: Boolean indicating whether to collect data from a sample or the entire population.
        - sample_size: Size of the sample to collect data from if sample is True.
        """
        # Select the population to collect data from
        if sample:
            population_to_collect = np.random.choice(host_population, sample_size, replace=False)
        else:
            population_to_collect = host_population
    
        # Shuffle the population to avoid the need for random sampling
        rnd.shuffle(population_to_collect)
        
        collected_data = []
        collected_vaccination_data = []
    
        # To measure vaccine efficacy we will gather data about the number of vaccinated hosts who get infected
        # along with the number of unvaccinated hosts that get infected
        vaccinated_hosts = []
        unvaccinated_hosts = []
    
        for h in population_to_collect:
            if not sample:
                # For vaccination data file, we will count the number of peopele with current vaccine immunity
                # This will exclude those who previously got the vaccine but the immunity waned.
                if h.vaccine is not None:
                    for vs in [get_strain_antigenic_name(s) for s in h.vaccine[0]]:
                        collected_vaccination_data.append((h.id, vs, t, h.get_age_category(), h.vaccine[2]))
            if len(h.prior_vaccinations) != 0:
                if len(vaccinated_hosts) < 1000:
                    vaccinated_hosts.append(h)
            else:
                if len(unvaccinated_hosts) < 1000:
                    unvaccinated_hosts.append(h)
            if h.isInfected():
                strain_str = [(path.get_strain_name(), path.is_severe, path.creation_time) for path in h.infecting_pathogen if not sample or not path.is_reassortant]
                for strain in strain_str:
                    collected_data.append((h.id, strain[0], t, h.get_age_category(), strain[1], strain[2], len(host_pop)))
                    
        # Only collect the vaccine efficacy data if we have vaccinated hosts
        if done_vaccinated:
            num_vaccinated = len(vaccinated_hosts)
            num_unvaccinated = len(unvaccinated_hosts)
            num_vaccinated_infected = 0
            num_unvaccinated_infected = 0
            num_vaccinated_infected_severe = 0
            num_unvaccinated_infected_severe = 0
            num_full_heterotypic = [0, 0]
            num_partial_heterotypic = [0, 0]
            num_homotypic = [0, 0]
    
            for vaccinated_host in vaccinated_hosts:
                if len(vaccinated_host.infections_with_vaccination) > 0:
                    num_vaccinated_infected += 1
                was_there_a_severe_infection = False
                was_there_a_full_heterotypic_infection = [False, False]
                was_there_a_partial_heterotypic_infection = [False, False]
                was_there_a_homotypic_infection = [False, False]
                for infecting_pathogen in vaccinated_host.infections_with_vaccination:     
                    index = 0           
                    if infecting_pathogen[0].is_severe:
                        index = 1
                        was_there_a_severe_infection = True
                    if infecting_pathogen[1] == PathogenMatch.HOMOTYPIC:
                        was_there_a_full_heterotypic_infection[index] = True
                    elif infecting_pathogen[1] == PathogenMatch.PARTIAL_HETERO:
                        was_there_a_partial_heterotypic_infection[index] = True
                    elif infecting_pathogen[1] == PathogenMatch.COMPLETE_HETERO:
                        was_there_a_homotypic_infection[index] = True
                    
                if was_there_a_severe_infection:
                    num_vaccinated_infected_severe += 1
                if was_there_a_full_heterotypic_infection[0]:
                    num_full_heterotypic[0] += 1
                if was_there_a_full_heterotypic_infection[1]:                
                    num_full_heterotypic[1] += 1
                if was_there_a_partial_heterotypic_infection[0]:
                    num_partial_heterotypic[0] += 1
                if was_there_a_partial_heterotypic_infection[1]:
                    num_partial_heterotypic[1] += 1
                if was_there_a_homotypic_infection[0]:
                    num_homotypic[0] += 1
                if was_there_a_homotypic_infection[1]:
                    num_homotypic[1] += 1
    
            for unvaccinated_host in unvaccinated_hosts:
                if len(unvaccinated_host.infections_without_vaccination) > 0:
                    num_unvaccinated_infected += 1
                was_there_a_severe_infection = False
                for infecting_pathogen in unvaccinated_host.infections_without_vaccination:                
                    if infecting_pathogen.is_severe:
                        was_there_a_severe_infection = True
                        break
                if was_there_a_severe_infection:
                    num_unvaccinated_infected_severe += 1
    
            with open(vaccine_efficacy_output_filename, "a", newline='') as outputfile:
                write = csv.writer(outputfile)
                write.writerow([t, num_vaccinated, num_unvaccinated, num_vaccinated_infected, num_vaccinated_infected_severe, num_unvaccinated_infected, num_unvaccinated_infected_severe,
                                num_homotypic[0], num_homotypic[1], num_partial_heterotypic[0], num_partial_heterotypic[1], num_full_heterotypic[0], num_full_heterotypic[1]]) 
    
        # Write collected data to the output file
        with open(output_filename, "a", newline='') as outputfile:
            writer = csv.writer(outputfile)
            writer.writerows(collected_data)
        if not sample:
            with open(vaccine_output_filename, "a", newline='') as outputfile:
                writer = csv.writer(outputfile)
                writer.writerows(collected_vaccination_data)
    
    
    def solve_quadratic(a, b, c):
        discriminant = b**2 - 4*a*c
        if discriminant >= 0:
            root1 = (-b + discriminant**0.5) / (2*a)
            root2 = (-b - discriminant**0.5) / (2*a)
            return tuple(sorted([root1, root2]))
        else:
            return "No real roots"
        
    def breakdown_vaccine_efficacy(ve, x):
        (r1, r2) = solve_quadratic(x, -(1+x), ve)
        if verbose: print(r1, r2)
        if r1 >= 0 and r1 <= 1:
            ve_s = r1
        elif r2 >= 0 and r2 <= 1:
            ve_s = r2
        else:
            print("No valid solution to the equation: x: %d, ve: %d. Solutions: %f %f" % (x, ve, r1, r2))
            exit(-1)
        ve_i = x * ve_s
        return (ve_i, ve_s)
    
    def get_probability_of_severe(pathogen_in, vaccine, immunity_count):
        if immunity_count >= 3:
            severity_probability = 0.18
        elif immunity_count == 2:
            severity_probability = 0.24
        elif immunity_count == 1:
            severity_probability = 0.23
        elif immunity_count == 0:
            severity_probability = 0.17
    
        if vaccine is not None:
            # print(vaccine[0][0], pathogen_in.strain)
            # Probability of severity also depends on the strain (homotypic/heterltypic/etc.)
            pathogen_strain_type = pathogen_in.match(vaccine[0][0])
            # Effectiveness of the vaccination depends on the number of doses
            if vaccine[2] == 1:
                ve_s = vaccine_efficacy_s_d1[pathogen_strain_type]
            elif vaccine[2] == 2:
                ve_s = vaccine_efficacy_s_d2[pathogen_strain_type]
            else:
                print("Unsupported vaccine dose")
                exit(-1)
            #  print(severity_probability * (1-ve_s), severity_probability, immunity_count)
            return severity_probability * (1-ve_s)
        else:
            return severity_probability
        
    
    ########## Set Parameters ##########
    N = 100000  # population size
    mu = 1.0/70.0     # average life span is 70 years
    gamma = 365/7  # 1/average infectious period (1/gamma =7 days)
    if waning_hypothesis == 1:
        omega = 365/273  # duration of immunity by infection= 39 weeks
    elif waning_hypothesis == 2:
        omega = 365/50  # duration of immunity by infection= 1 year
    elif waning_hypothesis == 3:
        omega = 365/100  # duration of immunity by infection= 2 years
    birth_rate = mu * 4
    
    cont = 365/1     # assumption
    timelimit = 40  #### 50 years 
    initialInfecteds = 10        ### for each strain infect initialInfecteds (with 2)
    
    reassortmentRate_GP = reassortment_rate
    
    if immunity_hypothesis == 1 or immunity_hypothesis == 4 or immunity_hypothesis == 5:
        partialCrossImmunityRate = 0
        completeHeterotypicImmunityrate = 0
    elif immunity_hypothesis == 2 :
        partialCrossImmunityRate = 1
        completeHeterotypicImmunityrate = 0
    elif immunity_hypothesis == 3:
        partialCrossImmunityRate = 0.5
        completeHeterotypicImmunityrate = 0
    elif immunity_hypothesis == 4:
        partialCrossImmunityRate = 0.95
        completeHeterotypicImmunityrate = 0.9
    elif immunity_hypothesis == 5:
        partialCrossImmunityRate = 0.95
        completeHeterotypicImmunityrate = 0.90
    elif immunity_hypothesis == 7:
        HomotypicImmunityRate = 0.95
        partialCrossImmunityRate = 0.90
        completeHeterotypicImmunityrate = 0.2
    elif immunity_hypothesis == 8:
        HomotypicImmunityRate = 0.9
        partialCrossImmunityRate = 0.5
        completeHeterotypicImmunityrate = 0
    elif immunity_hypothesis == 9:
        HomotypicImmunityRate = 0.9
        partialCrossImmunityRate = 0.45
        completeHeterotypicImmunityrate = 0.35
    elif immunity_hypothesis == 10:
        HomotypicImmunityRate = 0.8
        partialCrossImmunityRate = 0.45
        completeHeterotypicImmunityrate = 0.35
    else:
        print("No partial cross immunity rate for immunity hypothesis: ", immunity_hypothesis)
        exit(-1) 
    
    done_vaccinated = False
    vaccination_time =  20

    # Effectiveness of the vaccine first dose
    vaccine_efficacy_d1 = {
        PathogenMatch.HOMOTYPIC: 0.6,
        PathogenMatch.PARTIAL_HETERO: 0.45,
        PathogenMatch.COMPLETE_HETERO:0.15,
    }
    # Effectiveness of the vaccine second dose
    vaccine_efficacy_d2 = {
        PathogenMatch.HOMOTYPIC: 0.8,
        PathogenMatch.PARTIAL_HETERO: 0.65,
        PathogenMatch.COMPLETE_HETERO:0.35,
    }
    
    vaccine_efficacy_i_d1 = {}
    vaccine_efficacy_s_d1 = {}
    vaccine_efficacy_i_d2 = {}
    vaccine_efficacy_s_d2 = {}
    for (k, v) in vaccine_efficacy_d1.items():
        (ve_i, ve_s) = breakdown_vaccine_efficacy(v, ve_i_to_ve_s_ratio)
        vaccine_efficacy_i_d1[k] = ve_i
        vaccine_efficacy_s_d1[k] = ve_s
    for (k, v) in vaccine_efficacy_d2.items():
        (ve_i, ve_s) = breakdown_vaccine_efficacy(v, ve_i_to_ve_s_ratio)
        vaccine_efficacy_i_d2[k] = ve_i
        vaccine_efficacy_s_d2[k] = ve_s
    
    if verbose: print("VE_i: ", vaccine_efficacy_i_d1)
    if verbose: print("VE_s: ", vaccine_efficacy_s_d1)
    
    # Vaccination rates are derived based on the following formula
    vaccine_second_dose_rate = 0.8
    vaccine_first_dose_rate = math.sqrt(vaccine_second_dose_rate)
    if verbose: print("Vaccination - first dose rate: %s, second dose rate %s" % (vaccine_first_dose_rate, vaccine_second_dose_rate))
    
    vacinnation_single_dose_waning_rate = 365/273 #365/1273
    vacinnation_double_dose_waning_rate = 365/546 #365/2600
    # vacinnation_waning_lower_bound = 20 * 7 / 365.0
    
    total_strain_counts_vaccine = {}
    
    ### Tau leap parametes
    tau = 1/365.0
    
    numSegments = 4
    numNoneAgSegments = 2
    numAgSegments = numSegments - numNoneAgSegments
    #segmentVariants = [[i for i in range(1, 3)], [i for i in range(1, 3)], [i for i in range(1, 2)], [i for i in range(1, 2)]]     ## creating variats for the segments
    segmentVariants = [[1,2,3,4,9,11,12], [8,4,6], [i for i in range(1, 2)], [i for i in range(1, 2)]]
    segmentCombinations = [tuple(i) for i in itertools.product(*segmentVariants)]  # getting all possible combinations from a list of list
    rnd.shuffle(segmentCombinations)
    number_all_strains = len(segmentCombinations)
    initialSegmentCombinations = {(1,8,1,1): 100, (2,4,1,1): 100 ,(9,8,1,1): 100, (4,8,1,1): 100, (3,8,1,1): 100, (12,8,1,1): 100, (12,6,1,1): 100 ,(9,4,1,1): 100, (9,6,1,1): 100, (1,6,1,1): 100, (2,8,1,1): 100, (2,6,1,1): 100, (11,8,1,1): 100, (11,6,1,1): 100 ,(1,4,1,1): 100, (12,4,1,1): 100 }
    #initialSegmentCombinations = {(1,1,1,1): 500, (2,2,1,1): 10} #, (1,3,1,1): 3} #, (4,1,1,1): 3} #, (5,1,1,1): 10, (6,1,1,1): 10, (7,1,1,1): 10, (8,1,1,1): 10}

    num_initial_immune = 10000

    # Track the number of immune hosts(immunityCounts) in the host population
    immunityCounts = 0
    ReassortmentCount = 0
    pop_id = 0
    
    infected_pop = []
    pathogens_pop = []
    
    # for each strain track the number of hosts infected with it at current time: strainCount  
    strainCount = {}   # a defaultdict eliminates the need to check if a key exist in the the dictionary
    
    # reset simulation time for each replicate
    t = 0.0
    
    host_pop = [host(i) for i in range(N)]   # for each number in range of N, make a new Host object-> line37, i is the id.
    pop_id = N
    to_be_vaccinated_pop = [] 
    single_dose_vaccinated_pop = []
    
    for i in range(number_all_strains):
        strainCount[segmentCombinations[i]] = 0

    if verbose:
        if initial_immunity:
            print("Initial immunity is set to True")
        else:
            print("Initial immunity is set to False")

    ### infecting the initial infecteds
    for (initial_strain, num_infected) in initialSegmentCombinations.items():
        if initial_immunity:
            for j in range(num_initial_immune):
                h = rnd.choice(host_pop)
                h.immunity[initial_strain] = t
                immunityCounts += 1
        
        for j in range(num_infected):                     
            h = rnd.choice(host_pop)
            if not h.isInfected():
                infected_pop.append(h) 
            p = pathogen(False, t, host = h, strain = initial_strain)
            pathogens_pop.append(p)
            h.infecting_pathogen.append(p)
            strainCount[p.strain] += 1                       # count the strain of each strain. e.g. if p.strain contains strain1, then add 1 to strainCount of strain1
    if verbose: print(strainCount)
    
    initialize_files(strainCount)   
    
    tau_steps = 0
    t0 = time.time() # for us to track the time it takes to run the simulation
    last_data_colllected = 0
    data_collection_rate = 0.1
    
    for strain, count in strainCount.items():
        if strain[:numAgSegments] in total_strain_counts_vaccine:
            total_strain_counts_vaccine[strain[:numAgSegments]] += count
        else:
            total_strain_counts_vaccine[strain[:numAgSegments]] = count
    
    ########## run simulation ##########
    event_dict = sc.objdict(
        births=0,
        deaths=0,
        recoveries=0,
        contacts=0,
        wanings=0,
        reassortments=0,
        vaccine_dose_1_wanings=0,
        vaccine_dose_2_wanings=0,
    )
    while t<timelimit:
        if tau_steps % 10 == 0:
            if verbose is not False: print("Current time: %f (Number of steps = %d)" % (t, tau_steps))
            if verbose: print(strainCount)
    
        ### Every 100 steps, write the age distribution of the population to a file
        if tau_steps % 100 == 0:
            age_dict = {}
            for age_range in host.age_labels:
                age_dict[age_range] = 0
            for h in host_pop:
                age_dict[h.get_age_category()] += 1
            if verbose: print("Ages: ", age_dict)
            with open(age_outputfilename, "a", newline='') as outputfile:
                write = csv.writer(outputfile)
                write.writerow(["{:.2}".format(t)] + list(age_dict.values()))
        
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
        events = get_event_counts(len(host_pop), len(infected_pop), immunityCounts, tau, reassortmentRate_GP, len(single_dose_hosts), len(double_dose_hosts))
        births, deaths, recoveries, contacts, wanings, reassortments, vaccine_dose_1_wanings, vaccine_dose_2_wanings = events
        if verbose: print("t={}, births={}, deaths={}, recoveries={}, contacts={}, wanings={}, reassortments={}, waning_vaccine_d1={}, waning_vaccine_d2={}".format(t, births, deaths, recoveries, contacts, wanings, reassortments, vaccine_dose_1_wanings, vaccine_dose_2_wanings))
    
        # Parse into dict
        event_dict[:] += events
        
        # perform the events for the obtained counts
        birth_events(births, host_pop)
        reassortment_event(infected_pop, reassortments) # calling the function
        for _ in range(contacts):
            contact_event(infected_pop, host_pop, strainCount)
        death_event(deaths, infected_pop, host_pop, strainCount)
        recovery_event(recoveries, infected_pop, strainCount)    
        waning_event(host_pop, wanings)
        waning_vaccinations_first_dose(single_dose_hosts, vaccine_dose_1_wanings)
        waning_vaccinations_second_dose(double_dose_hosts, vaccine_dose_2_wanings)
        
        # Collect the total counts of strains at each time step to determine the most prevalent strain for vaccination
        if not done_vaccinated:
            for strain, count in strainCount.items():
                total_strain_counts_vaccine[strain[:numAgSegments]] += count
        
        # Administer the first dose of the vaccine
        # Vaccination strain is the most prevalent strain in the population before the vaccination starts
        if vaccine_hypothesis!=0 and (not done_vaccinated) and t >= vaccination_time:
            # Sort the strains by the number of hosts infected with it in the past
            # Pick the last one from the sorted list as the most prevalent strain
            vaccinated_strain = sorted(list(total_strain_counts_vaccine.keys()), key=lambda x: total_strain_counts_vaccine[x])[-1]
            # Select hosts under 6 weeks of age for vaccinate
            child_host_pop = [h for h in host_pop if t - h.bday <= 0.13 and t - h.bday >= 0.09]
            # Use the vaccination rate to determine the number of hosts to vaccinate
            vaccination_count = int(len(child_host_pop)*vaccine_first_dose_rate)            
            sample_population = rnd.sample(child_host_pop, vaccination_count)
            if verbose: print("Vaccinating with strain: ", vaccinated_strain, vaccination_count)
            if verbose: print("Number of people vaccinated: {} NUmber of people under 6 weeks: {}".format(len(sample_population), len(child_host_pop)))
            for h in sample_population:
                h.vaccinate(vaccinated_strain)
                single_dose_vaccinated_pop.append(h)
            done_vaccinated = True
        elif done_vaccinated:
            for child in to_be_vaccinated_pop:
                if t - child.bday >= 0.11:
                    child.vaccinate(vaccinated_strain)
                    to_be_vaccinated_pop.remove(child)
                    single_dose_vaccinated_pop.append(child)
    
        # Administer the second dose of the vaccine if first dose has already been administered.
        # The second dose is administered 6 weeks after the first dose with probability vaccine_second_dose_rate
        if done_vaccinated:
            while len(single_dose_vaccinated_pop) > 0:
                # If the first dose of the vaccine is older than 6 weeks then administer the second dose
                if t - single_dose_vaccinated_pop[0].vaccine[1] >= 0.11:
                    child = single_dose_vaccinated_pop.pop(0)
                    if rnd.random() < vaccine_second_dose_rate:
                        child.vaccinate(vaccinated_strain)
                else:
                    break
    
        if t >= last_data_colllected:
            collect_and_write_data(host_pop, sample_outputfilename, vaccinations_outputfilename, sample_vaccine_efficacy_output_filename, sample=True)
            collect_and_write_data(host_pop, infected_all_outputfilename, vaccinations_outputfilename, vaccine_efficacy_output_filename, sample=False)
            last_data_colllected += data_collection_rate
            
        with open(outputfilename, "a", newline='') as outputfile:
            write = csv.writer(outputfile)
            write.writerow([t] + list(strainCount.values()) + [ReassortmentCount])
    
        tau_steps += 1
        t+=tau
    
    t1 = time.time()
    total_time = t1-t0
    if verbose is not False: print("Time to run experiment: ", total_time)
    
    return event_dict


if __name__ == '__main__':
    events = main()