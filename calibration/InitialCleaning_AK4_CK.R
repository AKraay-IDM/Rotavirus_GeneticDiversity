##clean output data files
library(rstudioapi)
library(plyr)
library(dplyr)
setwd(dirname(rstudioapi::getActiveDocumentContext()$path))
dat<-read.csv('../results/rota_strains_infected_all_1_0.1_2_1_1_0_0.5_1.csv')
head(dat)

#Look at all years
dat$Strain3<-'Other'
dat$Strain3[which(dat$Strain=='G1P8A1B1')]<-'G1P8'
dat$Strain3[which(dat$Strain=='G2P4A1B1')]<-'G2P4'
dat$Strain3[which(dat$Strain=='G9P8A1B1')]<-'G9P8'

dat$Year<-floor(dat$CollectionTime)
YearlyGenoDat<-ddply(dat, .(Year, Strain3), summarize, Geno_cases=n_distinct(id))
YearlyCases<-ddply(dat, .(Year), summarize, All_cases=n_distinct(id), mean_pop=mean(PopulationSize))
CasesGeno<-merge(YearlyGenoDat, YearlyCases, by='Year')
CasesGeno$geno_prop<-CasesGeno$Geno_cases/CasesGeno$All_cases

#Subset to years 1-9 for now
initial8<-subset(dat, dat$CollectionTime<9 & dat$CollectionTime>1)

table(initial8$Strain)
initial8$Strain3<-'Other'
initial8$Strain3[which(initial8$Strain=='G1P8A1B1')]<-'G1P8'
initial8$Strain3[which(initial8$Strain=='G2P4A1B1')]<-'G2P4'
initial8$Strain3[which(initial8$Strain=='G9P8A1B1')]<-'G9P8'

GenoDist<-as.data.frame(table(initial8$Strain3))
names(GenoDist)<-c('Strain', 'Frequency')
total<-sum(GenoDist$Frequency)
GenoDist$Proportion<-GenoDist$Frequency/total
#This is formatted the same as the genotype calibration file, just need to decide on names but otherwise fine
#working out the case age distribution

#First making new age bins
initial8$AgeCat<-NA
initial8$AgeCat[which(initial8$Age %in% c('0-2', '2-4', '4-6', '6-12'))]<-'<1 y'
initial8$AgeCat[which(initial8$Age %in% c('12-24'))]<-'1-2 y'
initial8$AgeCat[which(initial8$Age %in% c('24-36', '36-48', '48-60'))]<-'2-5 y'
initial8$AgeCat[which(initial8$Age=='60+')]<-'>=5 y'

#Now, getting cases by age bin
cases_summary<-ddply(initial8, .(AgeCat, CollectionTime), summarize, Cases_age=n_distinct(id))
head(cases_summary)

#Now, getting total population by time point
#It's already pooled, so just need one observation per time point and then multiply by age bin
#Doing one per age bin so that the frame is set up to easily multiply
pop_pooled<-ddply(initial8, .(CollectionTime, AgeCat), function(x) head(x,1))
pop_pooled2<-pop_pooled[,c(3,7,10)]

#Multiplying by fraction of the population in each age bin
pop_pooled2$Pop_Age<-NA
pop_pooled2$Pop_Age[which(pop_pooled2$AgeCat=='<1 y')]<-pop_pooled2$PopulationSize*0.025 #
pop_pooled2$Pop_Age[which(pop_pooled2$AgeCat=='1-2 y')]<-pop_pooled2$PopulationSize*0.025 #
pop_pooled2$Pop_Age[which(pop_pooled2$AgeCat=='2-5 y')]<-pop_pooled2$PopulationSize*0.075 #
pop_pooled2$Pop_Age[which(pop_pooled2$AgeCat=='>=5 y')]<-pop_pooled2$PopulationSize*0.875 

#Merge incidence with the age and time appropriate denominator to get the ratio
AgeIncidence<-merge(cases_summary, pop_pooled2, by=c('AgeCat', 'CollectionTime'))
AgeIncidence$IR_100k<-(AgeIncidence$Cases_age/AgeIncidence$Pop_Age)*100000

#Get an average to calibrate to average pre-vaccine period
Incidence_dist<-ddply(AgeIncidence, .(AgeCat), summarize, meanIR=mean(IR_100k))
print(Incidence_dist)