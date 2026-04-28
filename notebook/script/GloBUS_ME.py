
# # Settings and Imports

# %%
#%% GENERAL SETTING & STATEMENTS
import pandas as pd
import numpy as np
import os
import ctypes     
import math
import warnings
from pathlib import Path

def GloBUS_cement_analysis(
    Scenario_Code,
    verbose=False,
    intensive_rate=0.93,
    lifetime_extention_rate=70/30,
    mi_factor_he=0.9,
    eol_recycle_reuse_rate=0.15,
):
    scenario_code = "".join(Scenario_Code).upper() if isinstance(Scenario_Code, (list, tuple)) else str(Scenario_Code).upper()
    if len(scenario_code) != 4 or any(c not in ("B", "H") for c in scenario_code):
        raise ValueError("Scenario_Code must be a 4-character code with only 'B' or 'H', e.g. 'BHBH'.")

    ME1, ME2, ME3, ME4 = scenario_code
    # set current directory
    this_file = Path(__file__).resolve()
    repo_root = this_file.parents[2]
    dir_path_GloBUS = repo_root / "1_GloBUS_submodule"
    output_dir = repo_root / "data" / "1_mfa"
    output_dir.mkdir(parents=True, exist_ok=True)

    old_cwd = Path.cwd()
    os.chdir(dir_path_GloBUS)
    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)
    warnings.filterwarnings("ignore", category=pd.errors.SettingWithCopyWarning)

    # Set general constants
    regions = 26        #26 IMAGE regions
    res_building_types = 4  #4 residential building types: detached, semi-detached, appartments & high-rise 
    area = 2            #2 areas: rural & urban
    inflation = 1.2423  # gdp/cap inflation correction between 2005 (IMAGE data) & 2016 (commercial calibration) according to https://www.bls.gov/data/inflation_calculator.htm

    # Set Flags for sensitivity analysis
    flag_alpha = 0      # switch for the sensitivity analysis on alpha, if 1 the maximum alpha is 10% above the maximum found in the data
    flag_ExpDec = 0     # switch to choose between Gompertz and Exponential Decay function for commercial floorspace demand (0 = Gompertz, 1 = Expdec)
    flag_Normal = 0     # switch to choose between Weibull and Normal lifetime distributions (0 = Weibull, 1 = Normal)

    #-----------------------------------------------------------
    # %%
    # Set Flags for scenario analysis
    # Set Flags for scenario analysis
    # 1. More Intensive Use of Space
    if ME1 == 'B':
        flag_intensive_use = 'BAU' # switch to choose between normal and extensive use of space (BAU = normal, HE = intensive)
    elif ME1 == 'H':
        flag_intensive_use = 'HE' # switch to choose between normal and extensive use of space (BAU = normal, HE = intensive)
    intensive_rate = float(intensive_rate) # 7% lower than normal use of space

    # 2. Lifetime Extention
    if ME2 == 'B':
        lifetime_extention = 'BAU' # switch to choose between normal and extended lifetime (BAU = normal, HE = extended)
    elif ME2 == 'H':
        lifetime_extention = 'HE'
    lifetime_extention_rate = float(lifetime_extention_rate) # 70yrs/30yrs longer lifetime

    # 3. Material Substitution
    if ME3 == 'B':
        material_subsitution = 'BAU' # switch to choose between normal and subsituted material (BAU = normal, HE = substitution)
    elif ME3 == 'H':
        material_subsitution = 'HE'
    MI_factor = float(mi_factor_he) # 10% lower than normal material intensity

    # 4. EoL Recycling and Reuse
    if ME4 == 'B':
        Eol_recycle_reuse = 'BAU' # switch to choose between normal and EoL recycling and reuse (BAU = normal, HE = EoL recycling and reuse)
    elif ME4 == 'H':
        Eol_recycle_reuse = 'HE'
    Eol_recycle_reuse_rate_value = float(eol_recycle_reuse_rate) # 15% normal EoL recycling and reuse rate
    #-----------------------------------------------------------
    # %%
    # Load Population, Floor area, and Service value added (SVA) Database csv-files
    pop = pd.read_csv('files_population/pop.csv', index_col = [0])                # Pop; unit: million of people; meaning: global population (over time, by region)             
    rurpop = pd.read_csv('files_population/rurpop.csv', index_col = [0])          # rurpop; unit: %; meaning: the share of people living in rural areas (over time, by region)
    housing_type = pd.read_csv('files_population/Housing_type.csv')               # Housing_type; unit: %; meaning: the share of the NUMBER OF PEOPLE living in a particular building type (by region & by area) 
    floorspace = pd.read_csv('files_floor_area/res_Floorspace.csv')               # Floorspace; unit: m2/capita; meaning: the average m2 per capita (over time, by region & area)
    floorspace = floorspace[floorspace.Region != regions + 1]                     # Remove empty region 27
    avg_m2_cap = pd.read_csv('files_floor_area/Average_m2_per_cap.csv')           # Avg_m2_cap; unit: m2/capita; meaning: average square meters per person (by region & area (rural/urban) & building type) 
    sva_pc_2005 = pd.read_csv('files_GDP/sva_pc.csv', index_col = [0])
    sva_pc = sva_pc_2005 * inflation                                              # we use the inflation corrected SVA to adjust for the fact that IMAGE provides gdp/cap in 2005 US$

    # load material density data csv-files
    # Preferred: dedicated cement-intensity file from extended workflow.
    # Fallback: infer a teaching-friendly cement proxy from available concrete intensities in the submodule.
    cement_file = Path('files_material_density/cement_materials_intensity_normal.csv')
    if cement_file.exists():
        building_materials = pd.read_csv(cement_file)
    else:
        res_materials = pd.read_csv('files_material_density/Building_materials.csv')
        com_materials = pd.read_csv('files_material_density/materials_commercial.csv')

        res_cement = res_materials[['Region', 'Building_type', 'concrete']].copy()
        res_cement = res_cement.rename(columns={'concrete': 'cement'})
        res_cement['Area_type'] = 'rural'
        res_cement_urban = res_cement.copy()
        res_cement_urban['Area_type'] = 'urban'

        com_cement = com_materials[['Region', 'Building_type', 'concrete']].copy()
        com_cement = com_cement.rename(columns={'concrete': 'cement'})
        com_cement['Area_type'] = 'commercial'

        building_materials = pd.concat([res_cement, res_cement_urban, com_cement], ignore_index=True)

    # Load fitted regression parameters for comercial floor area estimate
    if flag_alpha == 0:
        gompertz = pd.read_csv('files_floor_area//files_commercial/Gompertz_parameters.csv', index_col = [0])
    else:
        gompertz = pd.read_csv('files_floor_area//files_commercial/Gompertz_parameters_alpha.csv', index_col = [0])


    # # GloBUS Script
    # %%
    # Ensure full time series  for pop & rurpop (interpolation, some years are missing)
    rurpop2 = rurpop.reindex(list(range(1970,2061,1))).interpolate()
    pop2 = pop.reindex(list(range(1970,2061,1))).interpolate()

    # Remove 1st year, to ensure same Table size as floorspace data (from 1971)
    pop2 = pop2.iloc[1:]
    rurpop2 = rurpop2.iloc[1:]

    # pre-calculate urban population
    urbpop = 1 - rurpop2                                                           # urban population is 1 - the fraction of people living in rural areas (rurpop)

    # Restructure the tables to regions as columns; for floorspace
    floorspace_rur = floorspace.pivot(index = "t", columns = "Region", values = "Rural")
    floorspace_urb = floorspace.pivot(index = "t", columns = "Region", values = "Urban")

    # Restructuring for square meters (m2/cap)
    avg_m2_cap_urb = avg_m2_cap.loc[avg_m2_cap['Area'] == 'Urban'].drop(labels=['Area'], axis=1).T  # Remove area column & Transpose
    avg_m2_cap_urb.columns = list(map(int,avg_m2_cap_urb.iloc[0]))                      # name columns according to the row containing the region-labels
    avg_m2_cap_urb2 = avg_m2_cap_urb.drop(['Region'])                                 # Remove idle row 

    avg_m2_cap_rur = avg_m2_cap.loc[avg_m2_cap['Area'] == 'Rural'].drop(labels=['Area'], axis=1).T  # Remove area column & Transpose
    avg_m2_cap_rur.columns = list(map(int,avg_m2_cap_rur.iloc[0]))                      # name columns according to the row containing the region-labels
    avg_m2_cap_rur2 = avg_m2_cap_rur.drop(['Region'])                                 # Remove idle row 

    # Restructuring for the Housing types (% of population living in them)
    housing_type_urb = housing_type.loc[housing_type['Area'] == 'Urban'].drop(labels=['Area'], axis=1).T  # Remove area column & Transpose
    housing_type_urb.columns = list(map(int,housing_type_urb.iloc[0]))                      # name columns according to the row containing the region-labels
    housing_type_urb2 = housing_type_urb.drop(['Region'])                                 # Remove idle row 

    housing_type_rur = housing_type.loc[housing_type['Area'] == 'Rural'].drop(labels=['Area'], axis=1).T  # Remove area column & Transpose
    housing_type_rur.columns = list(map(int,housing_type_rur.iloc[0]))                      # name columns according to the row containing the region-labels
    housing_type_rur2 = housing_type_rur.drop(['Region'])                                 # Remove idle row 

    # %%
    #%% COMMERCIAL building space demand (stock) calculated from Gomperz curve (fitted, using separate regression model)

    # Select gompertz curve paramaters for the total commercial m2 demand (stock)
    alpha = gompertz['All']['a'] if flag_ExpDec == 0 else 25.601
    beta =  gompertz['All']['b'] if flag_ExpDec == 0 else 28.431
    gamma = gompertz['All']['c'] if flag_ExpDec == 0 else 0.0415

    # find the total commercial m2 stock (in Millions of m2)
    commercial_m2_cap = pd.DataFrame(index = range(1971,2061), columns = range(1,27))
    for year in range(1971,2061):
        for region in range(1,27):
            if flag_ExpDec == 0:
                commercial_m2_cap[region][year] = alpha * math.exp(-beta * math.exp((-gamma/1000) * sva_pc[str(region)][year]))
            else:
                commercial_m2_cap[region][year] = max(0.542, alpha - beta * math.exp((-gamma/1000) * sva_pc[str(region)][year]))

    # Subdivide the total across Offices, Retail+, Govt+ & Hotels+
    commercial_m2_cap_office = pd.DataFrame(index = range(1971,2061), columns = range(1,27))    # Offices
    commercial_m2_cap_retail = pd.DataFrame(index = range(1971,2061), columns = range(1,27))    # Retail & Warehouses
    commercial_m2_cap_hotels = pd.DataFrame(index = range(1971,2061), columns = range(1,27))    # Hotels & Restaurants
    commercial_m2_cap_govern = pd.DataFrame(index = range(1971,2061), columns = range(1,27))    # Hospitals, Education, Government & Transportation

    minimum_com_office = 25
    minimum_com_retail = 25
    minimum_com_hotels = 25
    minimum_com_govern = 25

    for year in range(1971,2061):
        for region in range(1,27):

            # get the square meter per capita floorspace for 4 commercial applications
            office = gompertz['Office']['a'] * math.exp(-gompertz['Office']['b'] * math.exp((-gompertz['Office']['c']/1000) * sva_pc[str(region)][year]))
            retail = gompertz['Retail+']['a'] * math.exp(-gompertz['Retail+']['b'] * math.exp((-gompertz['Retail+']['c']/1000) * sva_pc[str(region)][year]))
            hotels = gompertz['Hotels+']['a'] * math.exp(-gompertz['Hotels+']['b'] * math.exp((-gompertz['Hotels+']['c']/1000) * sva_pc[str(region)][year]))
            govern = gompertz['Govt+']['a'] * math.exp(-gompertz['Govt+']['b'] * math.exp((-gompertz['Govt+']['c']/1000) * sva_pc[str(region)][year]))

            #calculate minimum values for later use in historic tail(Region 20: China @ 134 $/cap SVA)
            minimum_com_office = office if office < minimum_com_office else minimum_com_office      
            minimum_com_retail = retail if retail < minimum_com_retail else minimum_com_retail
            minimum_com_hotels = hotels if hotels < minimum_com_hotels else minimum_com_hotels
            minimum_com_govern = govern if govern < minimum_com_govern else minimum_com_govern

            # Then use the ratio's to subdivide the total commercial floorspace into 4 categories      
            commercial_sum = office + retail + hotels + govern

            commercial_m2_cap_office[region][year] = commercial_m2_cap[region][year] * (office/commercial_sum)
            commercial_m2_cap_retail[region][year] = commercial_m2_cap[region][year] * (retail/commercial_sum)
            commercial_m2_cap_hotels[region][year] = commercial_m2_cap[region][year] * (hotels/commercial_sum)
            commercial_m2_cap_govern[region][year] = commercial_m2_cap[region][year] * (govern/commercial_sum)


    # %%
    #%% Add historic tail (1720-1970) + 100 yr initial -----------------------------------------------------------

    # load historic population development
    hist_pop = pd.read_csv('files_initial_stock/hist_pop.csv', index_col = [0])  # initial population as a percentage of the 1970 population; unit: %; according to the Maddison Project Database (MPD) 2018 (Groningen University)

    # Determine the historical average global trend in floorspace/cap  & the regional rural population share based on the last 10 years of IMAGE data
    floorspace_urb_trend_by_region = [0 for j in range(0,26)]
    floorspace_rur_trend_by_region = [0 for j in range(0,26)]
    rurpop_trend_by_region = [0 for j in range(0,26)]
    commercial_m2_cap_office_trend = [0 for j in range(0,26)]
    commercial_m2_cap_retail_trend = [0 for j in range(0,26)]
    commercial_m2_cap_hotels_trend = [0 for j in range(0,26)]
    commercial_m2_cap_govern_trend = [0 for j in range(0,26)]

    # For the RESIDENTIAL & COMMERCIAL floorspace: Derive the annual trend (in m2/cap) over the initial 10 years of IMAGE data
    for region in range(1,27):
        floorspace_urb_trend_by_year = [0 for i in range(0,10)]
        floorspace_rur_trend_by_year = [0 for i in range(0,10)]
        commercial_m2_cap_office_trend_by_year = [0 for j in range(0,10)]    
        commercial_m2_cap_retail_trend_by_year = [0 for i in range(0,10)]   
        commercial_m2_cap_hotels_trend_by_year = [0 for j in range(0,10)]
        commercial_m2_cap_govern_trend_by_year = [0 for i in range(0,10)]

        # Get the growth by year (for the first 10 years)
        for year in range(1970,1980):
            floorspace_urb_trend_by_year[year-1970] = floorspace_urb[region][year+1]/floorspace_urb[region][year+2]
            floorspace_rur_trend_by_year[year-1970] = floorspace_rur[region][year+1]/floorspace_rur[region][year+2]
            commercial_m2_cap_office_trend_by_year[year-1970] = commercial_m2_cap_office[region][year+1]/commercial_m2_cap_office[region][year+2]
            commercial_m2_cap_retail_trend_by_year[year-1970] = commercial_m2_cap_retail[region][year+1]/commercial_m2_cap_retail[region][year+2] 
            commercial_m2_cap_hotels_trend_by_year[year-1970] = commercial_m2_cap_hotels[region][year+1]/commercial_m2_cap_hotels[region][year+2]
            commercial_m2_cap_govern_trend_by_year[year-1970] = commercial_m2_cap_govern[region][year+1]/commercial_m2_cap_govern[region][year+2]

        rurpop_trend_by_region[region-1] = ((1 - (rurpop[str(region)][1980]/rurpop[str(region)][1970]))/10)*100
        floorspace_urb_trend_by_region[region-1] = sum(floorspace_urb_trend_by_year)/10
        floorspace_rur_trend_by_region[region-1] = sum(floorspace_rur_trend_by_year)/10
        commercial_m2_cap_office_trend[region-1] = sum(commercial_m2_cap_office_trend_by_year)/10
        commercial_m2_cap_retail_trend[region-1] = sum(commercial_m2_cap_retail_trend_by_year)/10
        commercial_m2_cap_hotels_trend[region-1] = sum(commercial_m2_cap_hotels_trend_by_year)/10
        commercial_m2_cap_govern_trend[region-1] = sum(commercial_m2_cap_govern_trend_by_year)/10

    # Average global annual decline in floorspace/cap in %, rural: 1%; urban 1.2%;  commercial: 1.26-2.18% /yr   
    floorspace_urb_trend_global = (1 - (sum(floorspace_urb_trend_by_region)/26))*100              # in % decrease per annum
    floorspace_rur_trend_global = (1 - (sum(floorspace_rur_trend_by_region)/26))*100              # in % decrease per annum
    commercial_m2_cap_office_trend_global = (1 - (sum(commercial_m2_cap_office_trend)/26))*100    # in % decrease per annum
    commercial_m2_cap_retail_trend_global = (1 - (sum(commercial_m2_cap_retail_trend)/26))*100    # in % decrease per annum
    commercial_m2_cap_hotels_trend_global = (1 - (sum(commercial_m2_cap_hotels_trend)/26))*100    # in % decrease per annum
    commercial_m2_cap_govern_trend_global = (1 - (sum(commercial_m2_cap_govern_trend)/26))*100    # in % decrease per annum

    # define historic floorspace (1820-1970) in m2/cap
    floorspace_urb_1820_1970 = pd.DataFrame(index = range(1820,1971), columns = floorspace_urb.columns)
    floorspace_rur_1820_1970 = pd.DataFrame(index = range(1820,1971), columns = floorspace_rur.columns)
    rurpop_1820_1970 = pd.DataFrame(index = range(1820,1971), columns = rurpop.columns)
    pop_1820_1970 = pd.DataFrame(index = range(1820,1971), columns = pop2.columns)
    commercial_m2_cap_office_1820_1970 = pd.DataFrame(index = range(1820,1971), columns = commercial_m2_cap_office.columns)
    commercial_m2_cap_retail_1820_1970 = pd.DataFrame(index = range(1820,1971), columns = commercial_m2_cap_retail.columns)
    commercial_m2_cap_hotels_1820_1970 = pd.DataFrame(index = range(1820,1971), columns = commercial_m2_cap_hotels.columns)
    commercial_m2_cap_govern_1820_1970 = pd.DataFrame(index = range(1820,1971), columns = commercial_m2_cap_govern.columns)

    # Find minumum or maximum values in the original IMAGE data (Just for residential, commercial minimum values have been calculated above)
    minimum_urb_fs = floorspace_urb.values.min()    # Region 20: China
    minimum_rur_fs = floorspace_rur.values.min()    # Region 20: China
    maximum_rurpop = rurpop.values.max()            # Region 9 : Eastern Africa

    # Calculate the actual values used between 1820 & 1970, given the trends & the min/max values
    for region in range(1,regions+1):
        for year in range(1820,1971):
            # MAX of 1) the MINimum value & 2) the calculated value
            floorspace_urb_1820_1970[region][year] = max(minimum_urb_fs, floorspace_urb[region][1971] * ((100-floorspace_urb_trend_global)/100)**(1971-year))  # single global value for average annual Decrease
            floorspace_rur_1820_1970[region][year] = max(minimum_rur_fs, floorspace_rur[region][1971] * ((100-floorspace_rur_trend_global)/100)**(1971-year))  # single global value for average annual Decrease
            commercial_m2_cap_office_1820_1970[region][year] = max(minimum_com_office, commercial_m2_cap_office[region][1971] * ((100-commercial_m2_cap_office_trend_global)/100)**(1971-year))  # single global value for average annual Decrease  
            commercial_m2_cap_retail_1820_1970[region][year] = max(minimum_com_retail, commercial_m2_cap_retail[region][1971] * ((100-commercial_m2_cap_retail_trend_global)/100)**(1971-year))  # single global value for average annual Decrease
            commercial_m2_cap_hotels_1820_1970[region][year] = max(minimum_com_hotels, commercial_m2_cap_hotels[region][1971] * ((100-commercial_m2_cap_hotels_trend_global)/100)**(1971-year))  # single global value for average annual Decrease
            commercial_m2_cap_govern_1820_1970[region][year] = max(minimum_com_govern, commercial_m2_cap_govern[region][1971] * ((100-commercial_m2_cap_govern_trend_global)/100)**(1971-year))  # single global value for average annual Decrease
            # MIN of 1) the MAXimum value & 2) the calculated value        
            rurpop_1820_1970[str(region)][year] = min(maximum_rurpop, rurpop[str(region)][1970] * ((100 + rurpop_trend_by_region[region - 1])/100)**(1970 - year))  # average annual INcrease by region
            # just add the tail to the population (no min/max & trend is pre-calculated in hist_pop)        
            pop_1820_1970[str(region)][year] = hist_pop[str(region)][year] * pop[str(region)][1970]

    urbpop_1820_1970 = 1 - rurpop_1820_1970

    # To avoid full model setup in 1820 (all required stock gets built in yr 1) we assume another tail that linearly increases to the 1820 value over a 100 year time period, so 1720 = 0
    floorspace_urb_1721_1820 = pd.DataFrame(index = range(1721,1820), columns = floorspace_urb.columns)
    floorspace_rur_1721_1820 = pd.DataFrame(index = range(1721,1820), columns = floorspace_rur.columns)
    rurpop_1721_1820 = pd.DataFrame(index = range(1721,1820), columns = rurpop.columns)
    urbpop_1721_1820 = pd.DataFrame(index = range(1721,1820), columns = urbpop.columns)
    pop_1721_1820 = pd.DataFrame(index = range(1721,1820), columns = pop2.columns)
    commercial_m2_cap_office_1721_1820 = pd.DataFrame(index = range(1721,1820), columns = commercial_m2_cap_office.columns)
    commercial_m2_cap_retail_1721_1820 = pd.DataFrame(index = range(1721,1820), columns = commercial_m2_cap_retail.columns)
    commercial_m2_cap_hotels_1721_1820 = pd.DataFrame(index = range(1721,1820), columns = commercial_m2_cap_hotels.columns)
    commercial_m2_cap_govern_1721_1820 = pd.DataFrame(index = range(1721,1820), columns = commercial_m2_cap_govern.columns)

    for region in range(1,27):
        for time in range(1721,1820):
            #                                                        MAX(0,...) Because of floating point deviations, leading to negative stock in some cases
            floorspace_urb_1721_1820[int(region)][time]            = max(0.0, floorspace_urb_1820_1970[int(region)][1820] - (floorspace_urb_1820_1970[int(region)][1820]/100)*(1820-time))
            floorspace_rur_1721_1820[int(region)][time]            = max(0.0, floorspace_rur_1820_1970[int(region)][1820] - (floorspace_rur_1820_1970[int(region)][1820]/100)*(1820-time))
            rurpop_1721_1820[str(region)][time]                    = max(0.0, rurpop_1820_1970[str(region)][1820] - (rurpop_1820_1970[str(region)][1820]/100)*(1820-time))
            urbpop_1721_1820[str(region)][time]                    = max(0.0, urbpop_1820_1970[str(region)][1820] - (urbpop_1820_1970[str(region)][1820]/100)*(1820-time))
            pop_1721_1820[str(region)][time]                       = max(0.0, pop_1820_1970[str(region)][1820] - (pop_1820_1970[str(region)][1820]/100)*(1820-time))
            commercial_m2_cap_office_1721_1820[int(region)][time]  = max(0.0, commercial_m2_cap_office_1820_1970[region][1820] - (commercial_m2_cap_office_1820_1970[region][1820]/100)*(1820-time))
            commercial_m2_cap_retail_1721_1820[int(region)][time]  = max(0.0, commercial_m2_cap_retail_1820_1970[region][1820] - (commercial_m2_cap_retail_1820_1970[region][1820]/100)*(1820-time))
            commercial_m2_cap_hotels_1721_1820[int(region)][time]  = max(0.0, commercial_m2_cap_hotels_1820_1970[region][1820] - (commercial_m2_cap_hotels_1820_1970[region][1820]/100)*(1820-time))
            commercial_m2_cap_govern_1721_1820[int(region)][time]  = max(0.0, commercial_m2_cap_govern_1820_1970[region][1820] - (commercial_m2_cap_govern_1820_1970[region][1820]/100)*(1820-time))

    # combine historic with IMAGE data here
    rurpop_tail                     = pd.concat([rurpop_1820_1970, rurpop2], axis=0)
    urbpop_tail                     = pd.concat([urbpop_1820_1970, urbpop], axis=0)
    pop_tail                        = pd.concat([pop_1820_1970, pop2], axis=0)
    floorspace_urb_tail             = pd.concat([floorspace_urb_1820_1970, floorspace_urb], axis=0)
    floorspace_rur_tail             = pd.concat([floorspace_rur_1820_1970, floorspace_rur], axis=0)
    commercial_m2_cap_office_tail   = pd.concat([commercial_m2_cap_office_1820_1970, commercial_m2_cap_office], axis=0)
    commercial_m2_cap_retail_tail   = pd.concat([commercial_m2_cap_retail_1820_1970, commercial_m2_cap_retail], axis=0)
    commercial_m2_cap_hotels_tail   = pd.concat([commercial_m2_cap_hotels_1820_1970, commercial_m2_cap_hotels], axis=0)
    commercial_m2_cap_govern_tail   = pd.concat([commercial_m2_cap_govern_1820_1970, commercial_m2_cap_govern], axis=0)

    rurpop_tail                     = pd.concat([rurpop_1721_1820, rurpop_1820_1970, rurpop2], axis=0)
    urbpop_tail                     = pd.concat([urbpop_1721_1820, urbpop_1820_1970, urbpop], axis=0)
    pop_tail                        = pd.concat([pop_1721_1820, pop_1820_1970, pop2], axis=0)
    floorspace_urb_tail             = pd.concat([floorspace_urb_1721_1820, floorspace_urb_1820_1970, floorspace_urb], axis=0)
    floorspace_rur_tail             = pd.concat([floorspace_rur_1721_1820, floorspace_rur_1820_1970, floorspace_rur], axis=0)
    commercial_m2_cap_office_tail   = pd.concat([commercial_m2_cap_office_1721_1820, commercial_m2_cap_office_1820_1970, commercial_m2_cap_office], axis=0)
    commercial_m2_cap_retail_tail   = pd.concat([commercial_m2_cap_retail_1721_1820, commercial_m2_cap_retail_1820_1970, commercial_m2_cap_retail], axis=0)
    commercial_m2_cap_hotels_tail   = pd.concat([commercial_m2_cap_hotels_1721_1820, commercial_m2_cap_hotels_1820_1970, commercial_m2_cap_hotels], axis=0)
    commercial_m2_cap_govern_tail   = pd.concat([commercial_m2_cap_govern_1721_1820, commercial_m2_cap_govern_1820_1970, commercial_m2_cap_govern], axis=0)



    # %%
    # make the avg_m2_cap_urb2 and avg_m2_cap_rur2 into 4 lists that can apply to the intensive rate during time changed
    new_index = range(1721, 2061)

    avg_m2_cap_rur2_0 = avg_m2_cap_rur2.iloc[[0], :]
    avg_m2_cap_rur2_0 = pd.concat([avg_m2_cap_rur2_0]*pop_tail.shape[0], ignore_index=True)
    avg_m2_cap_rur2_0 = avg_m2_cap_rur2_0.rename(index=dict(zip(avg_m2_cap_rur2_0.index, new_index)))
    avg_m2_cap_rur2_0.columns=pop_tail.columns

    avg_m2_cap_rur2_1 = avg_m2_cap_rur2.iloc[[1], :]
    avg_m2_cap_rur2_1 = pd.concat([avg_m2_cap_rur2_1]*pop_tail.shape[0], ignore_index=True)
    avg_m2_cap_rur2_1 = avg_m2_cap_rur2_1.rename(index=dict(zip(avg_m2_cap_rur2_1.index, new_index)))
    avg_m2_cap_rur2_1.columns=pop_tail.columns

    avg_m2_cap_rur2_2 = avg_m2_cap_rur2.iloc[[2], :]
    avg_m2_cap_rur2_2 = pd.concat([avg_m2_cap_rur2_2]*pop_tail.shape[0], ignore_index=True)
    avg_m2_cap_rur2_2 = avg_m2_cap_rur2_2.rename(index=dict(zip(avg_m2_cap_rur2_2.index, new_index)))
    avg_m2_cap_rur2_2.columns=pop_tail.columns

    avg_m2_cap_rur2_3 = avg_m2_cap_rur2.iloc[[3], :]
    avg_m2_cap_rur2_3 = pd.concat([avg_m2_cap_rur2_3]*pop_tail.shape[0], ignore_index=True)
    avg_m2_cap_rur2_3 = avg_m2_cap_rur2_3.rename(index=dict(zip(avg_m2_cap_rur2_3.index, new_index)))
    avg_m2_cap_rur2_3.columns=pop_tail.columns

    avg_m2_cap_urb2_0 = avg_m2_cap_urb2.iloc[[0], :]
    avg_m2_cap_urb2_0 = pd.concat([avg_m2_cap_urb2_0]*pop_tail.shape[0], ignore_index=True)
    avg_m2_cap_urb2_0 = avg_m2_cap_urb2_0.rename(index=dict(zip(avg_m2_cap_urb2_0.index, new_index)))
    avg_m2_cap_urb2_0.columns=pop_tail.columns

    avg_m2_cap_urb2_1 = avg_m2_cap_urb2.iloc[[1], :]
    avg_m2_cap_urb2_1 = pd.concat([avg_m2_cap_urb2_1]*pop_tail.shape[0], ignore_index=True)
    avg_m2_cap_urb2_1 = avg_m2_cap_urb2_1.rename(index=dict(zip(avg_m2_cap_urb2_1.index, new_index)))
    avg_m2_cap_urb2_1.columns=pop_tail.columns

    avg_m2_cap_urb2_2 = avg_m2_cap_urb2.iloc[[2], :]
    avg_m2_cap_urb2_2 = pd.concat([avg_m2_cap_urb2_2]*pop_tail.shape[0], ignore_index=True)
    avg_m2_cap_urb2_2 = avg_m2_cap_urb2_2.rename(index=dict(zip(avg_m2_cap_urb2_2.index, new_index)))
    avg_m2_cap_urb2_2.columns=pop_tail.columns

    avg_m2_cap_urb2_3 = avg_m2_cap_urb2.iloc[[3], :]
    avg_m2_cap_urb2_3 = pd.concat([avg_m2_cap_urb2_3]*pop_tail.shape[0], ignore_index=True)
    avg_m2_cap_urb2_3 = avg_m2_cap_urb2_3.rename(index=dict(zip(avg_m2_cap_urb2_3.index, new_index)))
    avg_m2_cap_urb2_3.columns=pop_tail.columns

    # %% [markdown]
    # ## ME1:More Intensive USE

    # %%
    # Create two arrays with different decreasing factors
    factor1 = np.linspace(1, 1, 301)
    factor2 = np.linspace(1, intensive_rate, 39)

    # Concatenate the two arrays
    intensive_rate_list = np.concatenate((factor1, factor2), axis=0)

    if flag_intensive_use == 'HE':
        # Apply intensive-use factor to all 26 regions.
        for region in map(str, range(1, 27)):
            avg_m2_cap_urb2_0[region] = avg_m2_cap_urb2_0[region] * intensive_rate_list
            avg_m2_cap_urb2_1[region] = avg_m2_cap_urb2_1[region] * intensive_rate_list
            avg_m2_cap_urb2_2[region] = avg_m2_cap_urb2_2[region] * intensive_rate_list
            avg_m2_cap_urb2_3[region] = avg_m2_cap_urb2_3[region] * intensive_rate_list
            avg_m2_cap_rur2_0[region] = avg_m2_cap_rur2_0[region] * intensive_rate_list
            avg_m2_cap_rur2_1[region] = avg_m2_cap_rur2_1[region] * intensive_rate_list
            avg_m2_cap_rur2_2[region] = avg_m2_cap_rur2_2[region] * intensive_rate_list
            avg_m2_cap_rur2_3[region] = avg_m2_cap_rur2_3[region] * intensive_rate_list

        for region in range(1, 27):
            floorspace_urb_tail[region] = floorspace_urb_tail[region] * intensive_rate_list
            floorspace_rur_tail[region] = floorspace_rur_tail[region] * intensive_rate_list
            commercial_m2_cap_office_tail[region] = commercial_m2_cap_office_tail[region] * intensive_rate_list
            commercial_m2_cap_retail_tail[region] = commercial_m2_cap_retail_tail[region] * intensive_rate_list
            commercial_m2_cap_hotels_tail[region] = commercial_m2_cap_hotels_tail[region] * intensive_rate_list
            commercial_m2_cap_govern_tail[region] = commercial_m2_cap_govern_tail[region] * intensive_rate_list
        if verbose:
            print('ME1: YES (intensive use)')
    elif flag_intensive_use == 'BAU':
        #do nothing
        if verbose:
            print('ME1: NO (BAU)')

    # %% [markdown]
    # ## Floor Area Stock based on population and PCA(per cap area)

    # %%
    #%% FLOOR AREA STOCK -----------------------------------------------------------

    # adjust the share for urban/rural only (shares in csv are as percantage of the total(Rur + Urb), we needed to adjust the urban shares to add up to 1, same for rural)
    housing_type_rur3 = housing_type_rur2/housing_type_rur2.sum()
    housing_type_urb3 = housing_type_urb2/housing_type_urb2.sum()

    # calculte the total rural/urban population (pop2 = millions of people, rurpop2 = % of people living in rural areas)
    people_rur = pd.DataFrame(rurpop_tail.values*pop_tail.values, columns = pop_tail.columns, index = pop_tail.index)
    people_urb = pd.DataFrame(urbpop_tail.values*pop_tail.values, columns = pop_tail.columns, index = pop_tail.index)

    # calculate the total number of people (urban/rural) BY HOUSING TYPE (the sum of det,sem,app & hig equals the total population e.g. people_rur)
    people_det_rur = pd.DataFrame(housing_type_rur3.iloc[0].values*people_rur.values, columns = people_rur.columns, index = people_rur.index)
    people_sem_rur = pd.DataFrame(housing_type_rur3.iloc[1].values*people_rur.values, columns = people_rur.columns, index = people_rur.index)
    people_app_rur = pd.DataFrame(housing_type_rur3.iloc[2].values*people_rur.values, columns = people_rur.columns, index = people_rur.index)
    people_hig_rur = pd.DataFrame(housing_type_rur3.iloc[3].values*people_rur.values, columns = people_rur.columns, index = people_rur.index)

    people_det_urb = pd.DataFrame(housing_type_urb3.iloc[0].values*people_urb.values, columns = people_urb.columns, index = people_urb.index)
    people_sem_urb = pd.DataFrame(housing_type_urb3.iloc[1].values*people_urb.values, columns = people_urb.columns, index = people_urb.index)
    people_app_urb = pd.DataFrame(housing_type_urb3.iloc[2].values*people_urb.values, columns = people_urb.columns, index = people_urb.index)
    people_hig_urb = pd.DataFrame(housing_type_urb3.iloc[3].values*people_urb.values, columns = people_urb.columns, index = people_urb.index)

    # calculate the total m2 (urban/rural) BY HOUSING TYPE (= nr. of people * OWN avg m2, so not based on IMAGE)
    m2_unadjusted_det_rur = pd.DataFrame(avg_m2_cap_rur2_0 * people_det_rur, columns = people_det_rur.columns, index = people_det_rur.index)
    m2_unadjusted_sem_rur = pd.DataFrame(avg_m2_cap_rur2_1 * people_sem_rur, columns = people_sem_rur.columns, index = people_sem_rur.index)
    m2_unadjusted_app_rur = pd.DataFrame(avg_m2_cap_rur2_2 * people_app_rur, columns = people_app_rur.columns, index = people_app_rur.index)
    m2_unadjusted_hig_rur = pd.DataFrame(avg_m2_cap_rur2_3 * people_hig_rur, columns = people_hig_rur.columns, index = people_hig_rur.index)

    m2_unadjusted_det_urb = pd.DataFrame(avg_m2_cap_urb2_0 * people_det_urb, columns = people_det_urb.columns, index = people_det_urb.index)
    m2_unadjusted_sem_urb = pd.DataFrame(avg_m2_cap_urb2_1 * people_sem_urb, columns = people_sem_urb.columns, index = people_sem_urb.index)
    m2_unadjusted_app_urb = pd.DataFrame(avg_m2_cap_urb2_2 * people_app_urb, columns = people_app_urb.columns, index = people_app_urb.index)
    m2_unadjusted_hig_urb = pd.DataFrame(avg_m2_cap_urb2_3 * people_hig_urb, columns = people_hig_urb.columns, index = people_hig_urb.index)

    # Define empty dataframes for m2 adjustments
    total_m2_adj_rur = pd.DataFrame(index = m2_unadjusted_det_rur.index, columns = m2_unadjusted_det_rur.columns)
    total_m2_adj_urb = pd.DataFrame(index = m2_unadjusted_det_urb.index, columns = m2_unadjusted_det_urb.columns)

    # Sum all square meters in Rural area
    for j in range(1721,2061,1):
        for i in range(1,27,1):
            total_m2_adj_rur.loc[j,str(i)] = m2_unadjusted_det_rur.loc[j,str(i)] + m2_unadjusted_sem_rur.loc[j,str(i)] + m2_unadjusted_app_rur.loc[j,str(i)] + m2_unadjusted_hig_rur.loc[j,str(i)]

    # Sum all square meters in Urban area
    for j in range(1721,2061,1):
        for i in range(1,27,1):
            total_m2_adj_urb.loc[j,str(i)] = m2_unadjusted_det_urb.loc[j,str(i)] + m2_unadjusted_sem_urb.loc[j,str(i)] + m2_unadjusted_app_urb.loc[j,str(i)] + m2_unadjusted_hig_urb.loc[j,str(i)]

    # average square meter per person implied by our OWN data
    avg_m2_cap_adj_rur = pd.DataFrame(total_m2_adj_rur.values / people_rur.values, columns = people_rur.columns, index = people_rur.index) 
    avg_m2_cap_adj_urb = pd.DataFrame(total_m2_adj_urb.values / people_urb.values, columns = people_urb.columns, index = people_urb.index)

    # factor to correct square meters per capita so that we respect the IMAGE data in terms of total m2, but we use our own distinction between Building types
    m2_cap_adj_fact_rur = pd.DataFrame(floorspace_rur_tail.values / avg_m2_cap_adj_rur.values, columns = floorspace_rur_tail.columns, index = floorspace_rur_tail.index)
    m2_cap_adj_fact_urb = pd.DataFrame(floorspace_urb_tail.values / avg_m2_cap_adj_urb.values, columns = floorspace_urb_tail.columns, index = floorspace_urb_tail.index)

    # All m2 by region (in millions), Building_type & year (using the correction factor, to comply with IMAGE avg m2/cap)
    m2_det_rur = pd.DataFrame(m2_unadjusted_det_rur.values * m2_cap_adj_fact_rur.values, columns = m2_cap_adj_fact_rur.columns, index = m2_cap_adj_fact_rur.index)
    m2_sem_rur = pd.DataFrame(m2_unadjusted_sem_rur.values * m2_cap_adj_fact_rur.values, columns = m2_cap_adj_fact_rur.columns, index = m2_cap_adj_fact_rur.index)
    m2_app_rur = pd.DataFrame(m2_unadjusted_app_rur.values * m2_cap_adj_fact_rur.values, columns = m2_cap_adj_fact_rur.columns, index = m2_cap_adj_fact_rur.index)
    m2_hig_rur = pd.DataFrame(m2_unadjusted_hig_rur.values * m2_cap_adj_fact_rur.values, columns = m2_cap_adj_fact_rur.columns, index = m2_cap_adj_fact_rur.index)

    m2_det_urb = pd.DataFrame(m2_unadjusted_det_urb.values * m2_cap_adj_fact_urb.values, columns = m2_cap_adj_fact_urb.columns, index = m2_cap_adj_fact_urb.index)
    m2_sem_urb = pd.DataFrame(m2_unadjusted_sem_urb.values * m2_cap_adj_fact_urb.values, columns = m2_cap_adj_fact_urb.columns, index = m2_cap_adj_fact_urb.index)
    m2_app_urb = pd.DataFrame(m2_unadjusted_app_urb.values * m2_cap_adj_fact_urb.values, columns = m2_cap_adj_fact_urb.columns, index = m2_cap_adj_fact_urb.index)
    m2_hig_urb = pd.DataFrame(m2_unadjusted_hig_urb.values * m2_cap_adj_fact_urb.values, columns = m2_cap_adj_fact_urb.columns, index = m2_cap_adj_fact_urb.index)

    # Add a checksum to see if calculations based on adjusted OWN avg m2 (by building type) now match the total m2 according to IMAGE. 
    m2_sum_rur_OWN = m2_det_rur + m2_sem_rur + m2_app_rur + m2_hig_rur
    m2_sum_rur_IMAGE = pd.DataFrame(floorspace_rur_tail.values*people_rur.values, columns = m2_sum_rur_OWN.columns, index = m2_sum_rur_OWN.index)
    m2_checksum = m2_sum_rur_OWN - m2_sum_rur_IMAGE
    if m2_checksum.sum().sum() > 0.0000001 or m2_checksum.sum().sum() < -0.0000001:
        ctypes.windll.user32.MessageBoxW(0, "IMAGE & OWN m2 sums do not match", "Warning", 1)

    # Total RESIDENTIAL square meters by region
    m2 = m2_det_rur + m2_sem_rur + m2_app_rur + m2_hig_rur + m2_det_urb + m2_sem_urb + m2_app_urb + m2_hig_urb

    # Total m2 for COMMERCIAL Buildings
    commercial_m2_office = pd.DataFrame(commercial_m2_cap_office_tail.values * pop_tail.values, columns = m2_cap_adj_fact_urb.columns, index = m2_cap_adj_fact_urb.index)
    commercial_m2_retail = pd.DataFrame(commercial_m2_cap_retail_tail.values * pop_tail.values, columns = m2_cap_adj_fact_urb.columns, index = m2_cap_adj_fact_urb.index)
    commercial_m2_hotels = pd.DataFrame(commercial_m2_cap_hotels_tail.values * pop_tail.values, columns = m2_cap_adj_fact_urb.columns, index = m2_cap_adj_fact_urb.index)
    commercial_m2_govern = pd.DataFrame(commercial_m2_cap_govern_tail.values * pop_tail.values, columns = m2_cap_adj_fact_urb.columns, index = m2_cap_adj_fact_urb.index)

    # Teaching workflow only needs cement inflow output; no stock export files.

    # %% [markdown]
    # ## Floor Area Inflow & Outflow

    # %%
    #%% FLOOR AREA INFLOW & OUTFLOW

    import sys 
    sys.path.append(str(dir_path_GloBUS))
    import dynamic_stock_model
    from dynamic_stock_model import DynamicStockModel as DSM
    idx = pd.IndexSlice   # needed for slicing multi-index

    # define a function for calculating the floor area inflow and outflow
    def inflow_outflown(shape, scale, stock, length):            # length is the number of years in the entire period
        out_oc_reg = pd.DataFrame(index = range(1721,2061), columns =  pd.MultiIndex.from_product([list(range(1,27)), list(range(1721,2061))]))  # Multi-index columns (region & years), to contain a matrix of years*years for each region
        out_i_reg = pd.DataFrame(index = range(1721,2061), columns = range(1,27))
        out_s_reg = pd.DataFrame(index = range(1721,2061), columns = range(1,27))
        out_o_reg = pd.DataFrame(index = range(1721,2061), columns = range(1,27))

        for region in range(1,27):
            shape_list = shape.loc[region]
            scale_list = scale.loc[region]

            if flag_Normal == 0:
                DSMforward = DSM(t = np.arange(0,length,1), s =  np.array(stock[region]), lt = {'Type': 'Weibull', 'Shape': np.array(shape_list), 'Scale': np.array(scale_list)})
            else:
                DSMforward = DSM(t = np.arange(0,length,1), s =  np.array(stock[region]), lt = {'Type': 'FoldNorm', 'Mean': np.array(shape_list), 'StdDev': np.array(scale_list)}) # shape & scale list are actually Mean & StDev here

            out_sc, out_oc, out_i = DSMforward.compute_stock_driven_model(NegativeInflowCorrect = True)

            out_i_reg[region] = out_i        
            out_oc[out_oc < 0] = 0 # remove negative outflow, replace by 0
            out_oc_reg.loc[:,idx[region,:]]  = out_oc

            # If you are only interested in the total outflow, you can sum the outflow by cohort
            out_o_reg[region] = out_oc.sum(axis = 1)
            out_o_reg_corr = out_o_reg._get_numeric_data()        
            out_o_reg_corr[out_o_reg_corr < 0] = 0            
            out_s_reg[region] = out_sc.sum(axis = 1) #Stock 

        return out_i_reg, out_oc_reg

    length = len(m2_hig_urb[1])  # = 340
    #nindex = np.arange(0,26)

    #% lifetime parameters (shape & scale)
    lifetimes = pd.read_csv(dir_path_GloBUS / 'files_lifetimes/lifetimes.csv')
    lifetimes_comm = pd.read_csv(dir_path_GloBUS / 'files_lifetimes/lifetimes_comm.csv')

    # separate shape from scale
    lifetimes_shape = lifetimes[['Region','Building_type','Area','Shape']]
    lifetimes_scale = lifetimes[['Region','Building_type','Area','Scale']]
    shape_comm = lifetimes_comm[['Region','Shape']]
    scale_comm = lifetimes_comm[['Region','Scale']]

    # generate time-series data structure
    for i in range(1721,2061):
        lifetimes_shape[i] = lifetimes_shape['Shape']
        lifetimes_scale[i] = lifetimes_scale['Scale']
        shape_comm[i] = shape_comm['Shape']
        scale_comm[i] = scale_comm['Scale']

    # parameters by building type    
    lifetimes_shape = lifetimes_shape.drop(['Shape'],axis = 1)
    lifetimes_scale = lifetimes_scale.drop(['Scale'],axis = 1)
    shape_comm = shape_comm.drop(['Shape'],axis = 1).set_index('Region')
    scale_comm = scale_comm.drop(['Scale'],axis = 1).set_index('Region')

    shape_det_rur = lifetimes_shape.loc[(lifetimes_shape['Area'] == 'Rural') & (lifetimes_shape['Building_type'] == 'Detached')].set_index('Region').drop(['Building_type', 'Area'],axis = 1)
    shape_sem_rur = lifetimes_shape.loc[(lifetimes_shape['Area'] == 'Rural') & (lifetimes_shape['Building_type'] == 'Semi-detached')].set_index('Region').drop(['Building_type', 'Area'],axis = 1)
    shape_app_rur = lifetimes_shape.loc[(lifetimes_shape['Area'] == 'Rural') & (lifetimes_shape['Building_type'] == 'Appartments')].set_index('Region').drop(['Building_type', 'Area'],axis = 1)
    shape_hig_rur = lifetimes_shape.loc[(lifetimes_shape['Area'] == 'Rural') & (lifetimes_shape['Building_type'] == 'High-rise')].set_index('Region').drop(['Building_type', 'Area'],axis = 1)

    shape_det_urb = lifetimes_shape.loc[(lifetimes_shape['Area'] == 'Urban') & (lifetimes_shape['Building_type'] == 'Detached')].set_index('Region').drop(['Building_type', 'Area'],axis = 1)
    shape_sem_urb = lifetimes_shape.loc[(lifetimes_shape['Area'] == 'Urban') & (lifetimes_shape['Building_type'] == 'Semi-detached')].set_index('Region').drop(['Building_type', 'Area'],axis = 1)
    shape_app_urb = lifetimes_shape.loc[(lifetimes_shape['Area'] == 'Urban') & (lifetimes_shape['Building_type'] == 'Appartments')].set_index('Region').drop(['Building_type', 'Area'],axis = 1)
    shape_hig_urb = lifetimes_shape.loc[(lifetimes_shape['Area'] == 'Urban') & (lifetimes_shape['Building_type'] == 'High-rise')].set_index('Region').drop(['Building_type', 'Area'],axis = 1)

    scale_det_rur = lifetimes_scale.loc[(lifetimes_scale['Area'] == 'Rural') & (lifetimes_scale['Building_type'] == 'Detached')].set_index('Region').drop(['Building_type', 'Area'],axis = 1)
    scale_sem_rur = lifetimes_scale.loc[(lifetimes_scale['Area'] == 'Rural') & (lifetimes_scale['Building_type'] == 'Semi-detached')].set_index('Region').drop(['Building_type', 'Area'],axis = 1)
    scale_app_rur = lifetimes_scale.loc[(lifetimes_scale['Area'] == 'Rural') & (lifetimes_scale['Building_type'] == 'Appartments')].set_index('Region').drop(['Building_type', 'Area'],axis = 1)
    scale_hig_rur = lifetimes_scale.loc[(lifetimes_scale['Area'] == 'Rural') & (lifetimes_scale['Building_type'] == 'High-rise')].set_index('Region').drop(['Building_type', 'Area'],axis = 1)

    scale_det_urb = lifetimes_scale.loc[(lifetimes_scale['Area'] == 'Urban') & (lifetimes_scale['Building_type'] == 'Detached')].set_index('Region').drop(['Building_type', 'Area'],axis = 1)
    scale_sem_urb = lifetimes_scale.loc[(lifetimes_scale['Area'] == 'Urban') & (lifetimes_scale['Building_type'] == 'Semi-detached')].set_index('Region').drop(['Building_type', 'Area'],axis = 1)
    scale_app_urb = lifetimes_scale.loc[(lifetimes_scale['Area'] == 'Urban') & (lifetimes_scale['Building_type'] == 'Appartments')].set_index('Region').drop(['Building_type', 'Area'],axis = 1)
    scale_hig_urb = lifetimes_scale.loc[(lifetimes_scale['Area'] == 'Urban') & (lifetimes_scale['Building_type'] == 'High-rise')].set_index('Region').drop(['Building_type', 'Area'],axis = 1)


    # %% [markdown]
    # ## ME2: Lifetime Extension

    # %%
    def lifetime_extention_func(input_scale, scale_multiplier):
        for region in range(1, 27):
            growth_values = np.linspace(input_scale.loc[region, 2020], input_scale.loc[region, 2060]*scale_multiplier, num=41)
            input_scale.loc[region, 2021:2060] = growth_values[1:]
    # apply lifetime extention
    if lifetime_extention == 'HE':
        scale_multiplier = lifetime_extention_rate
        lifetime_extention_func(shape_comm, scale_multiplier)
        lifetime_extention_func(scale_det_rur, scale_multiplier)
        lifetime_extention_func(scale_sem_rur, scale_multiplier)
        lifetime_extention_func(scale_app_rur, scale_multiplier)
        lifetime_extention_func(scale_hig_rur, scale_multiplier)
        lifetime_extention_func(scale_det_urb, scale_multiplier)
        lifetime_extention_func(scale_sem_urb, scale_multiplier)
        lifetime_extention_func(scale_app_urb, scale_multiplier)
        lifetime_extention_func(scale_hig_urb, scale_multiplier)
        if verbose:
            print('ME2: YES (lifetime extention)')
    elif lifetime_extention == 'BAU':
        if verbose:
            print('ME2: NO (BAU)')

    # %%
    # call the defined model to calculate inflow & outflow based on stock & lifetime
    m2_det_rur_i, m2_det_rur_oc = inflow_outflown(shape_det_rur, scale_det_rur, m2_det_rur, length)
    m2_sem_rur_i, m2_sem_rur_oc = inflow_outflown(shape_sem_rur, scale_sem_rur, m2_sem_rur, length)
    m2_app_rur_i, m2_app_rur_oc = inflow_outflown(shape_app_rur, scale_app_rur, m2_app_rur, length)
    m2_hig_rur_i, m2_hig_rur_oc = inflow_outflown(shape_hig_rur, scale_hig_rur, m2_hig_rur, length)

    m2_det_urb_i, m2_det_urb_oc = inflow_outflown(shape_det_urb, scale_det_urb, m2_det_urb, length)
    m2_sem_urb_i, m2_sem_urb_oc = inflow_outflown(shape_sem_urb, scale_sem_urb, m2_sem_urb, length)
    m2_app_urb_i, m2_app_urb_oc = inflow_outflown(shape_app_urb, scale_app_urb, m2_app_urb, length)
    m2_hig_urb_i, m2_hig_urb_oc = inflow_outflown(shape_hig_urb, scale_hig_urb, m2_hig_urb, length)

    m2_office_i, m2_office_oc = inflow_outflown(shape_comm, scale_comm, commercial_m2_office, length)
    m2_retail_i, m2_retail_oc = inflow_outflown(shape_comm, scale_comm, commercial_m2_retail, length)
    m2_hotels_i, m2_hotels_oc = inflow_outflown(shape_comm, scale_comm, commercial_m2_hotels, length)
    m2_govern_i, m2_govern_oc = inflow_outflown(shape_comm, scale_comm, commercial_m2_govern, length)

    # total MILLIONS of square meters inflow
    m2_res_i = m2_det_rur_i + m2_sem_rur_i + m2_app_rur_i + m2_hig_rur_i + m2_det_urb_i + m2_sem_urb_i + m2_app_urb_i + m2_hig_urb_i
    m2_comm_i = m2_office_i + m2_retail_i + m2_hotels_i + m2_govern_i


    # %% [markdown]
    # ## Smooth the Curve

    # %%
    # even using 9 years
    def even_curve(df):
        new_df = pd.DataFrame(columns=df.columns, index=df.index).fillna(0)
        for i,row in df.iterrows():
            for region in range(1, 27):
                value_before = row[region]
                step_size = value_before / 9
                if i == 1721:
                    new_df.loc[i:i+4, region] += step_size
                elif i == 1722:
                    new_df.loc[i-1:i+4, region] += step_size
                elif i == 1723:
                    new_df.loc[i-2:i+4, region] += step_size
                elif i == 1724:
                    new_df.loc[i-3:i+4, region] += step_size
                elif i == 2057:
                    new_df.loc[i-4:i+3, region] += step_size
                elif i == 2058:
                    new_df.loc[i-4:i+2, region] += step_size
                elif i == 2059:
                    new_df.loc[i-4:i+1, region] += step_size
                elif i == 2060:
                    new_df.loc[i-4:i, region] += step_size
                else:
                    new_df.loc[i-4:i+4, region] += step_size
        # readjust the last years
        for region in range(1, 27):
            new_df.loc[2060, region] = new_df.loc[2060, region] * 9/5
            new_df.loc[2059, region] = new_df.loc[2059, region] * 9/6
            new_df.loc[2058, region] = new_df.loc[2058, region] * 9/7
            new_df.loc[2057, region] = new_df.loc[2057, region] * 9/8
        return new_df

    m2_total = m2_det_rur_i + m2_sem_rur_i + m2_app_rur_i + m2_hig_rur_i + m2_det_urb_i + m2_sem_urb_i + m2_app_urb_i + m2_hig_urb_i + m2_office_i + m2_retail_i + m2_hotels_i + m2_govern_i
    #m2_total[20].plot()

    m2_det_rur_i = even_curve(m2_det_rur_i)
    m2_sem_rur_i = even_curve(m2_sem_rur_i)
    m2_app_rur_i = even_curve(m2_app_rur_i)
    m2_hig_rur_i = even_curve(m2_hig_rur_i)
    m2_det_urb_i = even_curve(m2_det_urb_i)
    m2_sem_urb_i = even_curve(m2_sem_urb_i)
    m2_app_urb_i = even_curve(m2_app_urb_i)
    m2_hig_urb_i = even_curve(m2_hig_urb_i)
    m2_office_i = even_curve(m2_office_i)
    m2_retail_i = even_curve(m2_retail_i)
    m2_hotels_i = even_curve(m2_hotels_i)
    m2_govern_i = even_curve(m2_govern_i)

    m2_total_new = m2_det_rur_i + m2_sem_rur_i + m2_app_rur_i + m2_hig_rur_i + m2_det_urb_i + m2_sem_urb_i + m2_app_urb_i + m2_hig_urb_i + m2_office_i + m2_retail_i + m2_hotels_i + m2_govern_i
    #m2_total_new[20].plot()

    # %% [markdown]
    # ## Material Inflow & Outflow

    # %%
    #%% MATERIAL INTENSITY RESTRUCTURING (to become consistent with floor area dataset)-----------------------------------------------------------
    # separate different materials
    building_materials_cement = building_materials[['Region','Area_type','Building_type','cement']]

    # generate time-series data structure
    for i in range(1721,2061):
        building_materials_cement[i] = building_materials_cement['cement']


    building_materials_cement = building_materials_cement.drop(['cement'],axis = 1)

    # cement intensity
    material_cement_urb_det = building_materials_cement.loc[(building_materials_cement['Building_type']=='Detached')&(building_materials_cement['Area_type']=='urban')].set_index('Region').drop(['Building_type','Area_type'],axis = 1).T.set_index(m2_det_rur.index)
    material_cement_urb_sem = building_materials_cement.loc[(building_materials_cement['Building_type']=='Semi-detached')&(building_materials_cement['Area_type']=='urban')].set_index('Region').drop(['Building_type','Area_type'],axis = 1).T.set_index(m2_sem_rur.index)
    material_cement_urb_app = building_materials_cement.loc[(building_materials_cement['Building_type']=='Appartments')&(building_materials_cement['Area_type']=='urban')].set_index('Region').drop(['Building_type','Area_type'],axis = 1).T.set_index(m2_app_rur.index)
    material_cement_urb_hig = building_materials_cement.loc[(building_materials_cement['Building_type']=='High-rise')&(building_materials_cement['Area_type']=='urban')].set_index('Region').drop(['Building_type','Area_type'],axis = 1).T.set_index(m2_hig_rur.index)

    material_cement_rur_det = building_materials_cement.loc[(building_materials_cement['Building_type']=='Detached')&(building_materials_cement['Area_type']=='rural')].set_index('Region').drop(['Building_type','Area_type'],axis = 1).T.set_index(m2_det_rur.index)
    material_cement_rur_sem = building_materials_cement.loc[(building_materials_cement['Building_type']=='Semi-detached')&(building_materials_cement['Area_type']=='rural')].set_index('Region').drop(['Building_type','Area_type'],axis = 1).T.set_index(m2_sem_rur.index)
    material_cement_rur_app = building_materials_cement.loc[(building_materials_cement['Building_type']=='Appartments')&(building_materials_cement['Area_type']=='rural')].set_index('Region').drop(['Building_type','Area_type'],axis = 1).T.set_index(m2_app_rur.index)
    material_cement_rur_hig = building_materials_cement.loc[(building_materials_cement['Building_type']=='High-rise')&(building_materials_cement['Area_type']=='rural')].set_index('Region').drop(['Building_type','Area_type'],axis = 1).T.set_index(m2_hig_rur.index)

    materials_cement_office = building_materials_cement.loc[(building_materials_cement['Building_type']=='Offices')&(building_materials_cement['Area_type']=='commercial')].set_index('Region').drop(['Building_type','Area_type'],axis = 1).T.set_index(commercial_m2_office.index)
    materials_cement_retail = building_materials_cement.loc[(building_materials_cement['Building_type']=='Retail+')&(building_materials_cement['Area_type']=='commercial')].set_index('Region').drop(['Building_type','Area_type'],axis = 1).T.set_index(commercial_m2_retail.index)
    materials_cement_hotels = building_materials_cement.loc[(building_materials_cement['Building_type']=='Hotels+')&(building_materials_cement['Area_type']=='commercial')].set_index('Region').drop(['Building_type','Area_type'],axis = 1).T.set_index(commercial_m2_hotels.index)
    materials_cement_govern = building_materials_cement.loc[(building_materials_cement['Building_type']=='Govt+')&(building_materials_cement['Area_type']=='commercial')].set_index('Region').drop(['Building_type','Area_type'],axis = 1).T.set_index(commercial_m2_govern.index)


    # %%
    #%% Material inflow & outflow
    #% Material inflow (Millions of kgs = *1000 tons)


    ## ME3: Material Intensity Efficiency
    # Construct the material intensity factor (MI_factor) as a list

    if material_subsitution == 'BAU':
        MI_factor = 1 
        if verbose:
            print('ME3: NO (BAU)')
    elif material_subsitution == 'HE':    
        MI_factor = float(mi_factor_he) # 10% less material intensity
        if verbose:
            print('ME3: YES (material intensity efficiency)')

    MI_factor_list_20to60 = np.linspace(1, MI_factor, num=41)
    MI_factor_list = np.concatenate((np.ones(299), MI_factor_list_20to60))

    # Repeat MI_factor_list for 26 regions and reshape it to match the shape of the DataFrame
    MI_factor_df = pd.DataFrame(np.tile(MI_factor_list, (26, 1)).T, index=material_cement_rur_det.index)

    # Create a dictionary mapping the old column names to the new ones
    column_mapping = {old: new for old, new in zip(MI_factor_df.columns, range(1, 27))}

    # Rename the columns
    MI_factor_df = MI_factor_df.rename(columns=column_mapping)

    # modify the material intensity factor by the MI_factor
    material_cement_rur_det = material_cement_rur_det.multiply(MI_factor_df)
    material_cement_rur_sem = material_cement_rur_sem.multiply(MI_factor_df)
    material_cement_rur_app = material_cement_rur_app.multiply(MI_factor_df)
    material_cement_rur_hig = material_cement_rur_hig.multiply(MI_factor_df)

    material_cement_urb_det = material_cement_urb_det.multiply(MI_factor_df)
    material_cement_urb_sem = material_cement_urb_sem.multiply(MI_factor_df)
    material_cement_urb_app = material_cement_urb_app.multiply(MI_factor_df)
    material_cement_urb_hig = material_cement_urb_hig.multiply(MI_factor_df)

    materials_cement_office = materials_cement_office.multiply(MI_factor_df)
    materials_cement_retail = materials_cement_retail.multiply(MI_factor_df)
    materials_cement_hotels = materials_cement_hotels.multiply(MI_factor_df)
    materials_cement_govern = materials_cement_govern.multiply(MI_factor_df)

    # produce the material inflow by the modified material intensity factor
    kg_det_rur_cement_i = m2_det_rur_i.multiply(material_cement_rur_det)
    kg_sem_rur_cement_i = m2_sem_rur_i.multiply(material_cement_rur_sem)
    kg_app_rur_cement_i = m2_app_rur_i.multiply(material_cement_rur_app)
    kg_hig_rur_cement_i = m2_hig_rur_i.multiply(material_cement_rur_hig)

    kg_det_urb_cement_i = m2_det_urb_i.multiply(material_cement_urb_det)
    kg_sem_urb_cement_i = m2_sem_urb_i.multiply(material_cement_urb_sem)
    kg_app_urb_cement_i = m2_app_urb_i.multiply(material_cement_urb_app)
    kg_hig_urb_cement_i = m2_hig_urb_i.multiply(material_cement_urb_hig)

    kg_office_cement_i = m2_office_i.multiply(materials_cement_office)
    kg_retail_cement_i = m2_retail_i.multiply(materials_cement_retail)
    kg_hotels_cement_i = m2_hotels_i.multiply(materials_cement_hotels)
    kg_govern_cement_i = m2_govern_i.multiply(materials_cement_govern)


    #% Material outflow (Millions of kgs = *1000 tons)
    # first define a function for calculating the material outflow by cohort
    def material_outflow(m2_outflow_cohort,material_density):
        emp = []
        for i in range(0,26):
            md = material_density.iloc[:,i]
            m2 = m2_outflow_cohort.loc[:,(i+1,1721):(i+1,2060)]
            m2.columns = md.index
            material_outflow_cohort =  m2*md
            material_outflow_cohort_sum = material_outflow_cohort.sum(1)
            emp.append(material_outflow_cohort_sum)
        result = pd.DataFrame(emp)
        result.index = range(1, 27)
        return result.T

    # cement outflow
    kg_det_rur_cement_o = material_outflow(m2_det_rur_oc, material_cement_rur_det)
    kg_sem_rur_cement_o = material_outflow(m2_sem_rur_oc, material_cement_rur_sem)
    kg_app_rur_cement_o = material_outflow(m2_app_rur_oc, material_cement_rur_app)
    kg_hig_rur_cement_o = material_outflow(m2_hig_rur_oc, material_cement_rur_hig)

    kg_det_urb_cement_o = material_outflow(m2_det_urb_oc, material_cement_urb_det)
    kg_sem_urb_cement_o = material_outflow(m2_sem_urb_oc, material_cement_urb_sem)
    kg_app_urb_cement_o = material_outflow(m2_app_urb_oc, material_cement_urb_app)
    kg_hig_urb_cement_o = material_outflow(m2_hig_urb_oc, material_cement_urb_hig)

    kg_office_cement_o = material_outflow(m2_office_oc, materials_cement_office)
    kg_retail_cement_o = material_outflow(m2_retail_oc, materials_cement_retail)
    kg_hotels_cement_o = material_outflow(m2_hotels_oc, materials_cement_hotels)
    kg_govern_cement_o = material_outflow(m2_govern_oc, materials_cement_govern)


    ## ME4: Reuse the cement from demolished buildings

    RR_rate_20to60 = np.linspace(0, Eol_recycle_reuse_rate_value, num=41)
    Eol_recycle_reuse_rate = np.concatenate((np.zeros(299), RR_rate_20to60))
    Eol_recycle_reuse_rate = pd.Series(Eol_recycle_reuse_rate)


    # Define a function for calculating the material outflow by cohort
    if Eol_recycle_reuse == 'HE': 
        for region in range(1, 27):
            kg_det_rur_cement_o[region] = kg_det_rur_cement_o[region].values * (1 - Eol_recycle_reuse_rate).values
            kg_sem_rur_cement_o[region] = kg_sem_rur_cement_o[region].values * (1 - Eol_recycle_reuse_rate).values
            kg_app_rur_cement_o[region] = kg_app_rur_cement_o[region].values * (1 - Eol_recycle_reuse_rate).values
            kg_hig_rur_cement_o[region] = kg_hig_rur_cement_o[region].values * (1 - Eol_recycle_reuse_rate).values

            kg_det_urb_cement_o[region] = kg_det_urb_cement_o[region].values * (1 - Eol_recycle_reuse_rate).values
            kg_sem_urb_cement_o[region] = kg_sem_urb_cement_o[region].values * (1 - Eol_recycle_reuse_rate).values
            kg_app_urb_cement_o[region] = kg_app_urb_cement_o[region].values * (1 - Eol_recycle_reuse_rate).values
            kg_hig_urb_cement_o[region] = kg_hig_urb_cement_o[region].values * (1 - Eol_recycle_reuse_rate).values

            kg_office_cement_o[region] = kg_office_cement_o[region].values * (1 - Eol_recycle_reuse_rate).values
            kg_retail_cement_o[region] = kg_retail_cement_o[region].values * (1 - Eol_recycle_reuse_rate).values
            kg_hotels_cement_o[region] = kg_hotels_cement_o[region].values * (1 - Eol_recycle_reuse_rate).values
            kg_govern_cement_o[region] = kg_govern_cement_o[region].values * (1 - Eol_recycle_reuse_rate).values

            kg_det_rur_cement_i[region] = kg_det_rur_cement_i[region].values - kg_det_rur_cement_o[region].values * Eol_recycle_reuse_rate.values
            kg_sem_rur_cement_i[region] = kg_sem_rur_cement_i[region].values - kg_sem_rur_cement_o[region].values * Eol_recycle_reuse_rate.values
            kg_app_rur_cement_i[region] = kg_app_rur_cement_i[region].values - kg_app_rur_cement_o[region].values * Eol_recycle_reuse_rate.values
            kg_hig_rur_cement_i[region] = kg_hig_rur_cement_i[region].values - kg_hig_rur_cement_o[region].values * Eol_recycle_reuse_rate.values

            kg_det_urb_cement_i[region] = kg_det_urb_cement_i[region].values - kg_det_urb_cement_o[region].values * Eol_recycle_reuse_rate.values
            kg_sem_urb_cement_i[region] = kg_sem_urb_cement_i[region].values - kg_sem_urb_cement_o[region].values * Eol_recycle_reuse_rate.values
            kg_app_urb_cement_i[region] = kg_app_urb_cement_i[region].values - kg_app_urb_cement_o[region].values * Eol_recycle_reuse_rate.values
            kg_hig_urb_cement_i[region] = kg_hig_urb_cement_i[region].values - kg_hig_urb_cement_o[region].values * Eol_recycle_reuse_rate.values

            kg_office_cement_i[region] = kg_office_cement_i[region].values - kg_office_cement_o[region].values * Eol_recycle_reuse_rate.values
            kg_retail_cement_i[region] = kg_retail_cement_i[region].values - kg_retail_cement_o[region].values * Eol_recycle_reuse_rate.values
            kg_hotels_cement_i[region] = kg_hotels_cement_i[region].values - kg_hotels_cement_o[region].values * Eol_recycle_reuse_rate.values
            kg_govern_cement_i[region] = kg_govern_cement_i[region].values - kg_govern_cement_o[region].values * Eol_recycle_reuse_rate.values
        if verbose:
            print('ME4: YES (End of life recycle and reuse)')
    elif Eol_recycle_reuse == 'BAU':
        if verbose:
            print('ME4: NO (BAU)')


    # %% [markdown]
    # ## CSV output

    # %%
    #%% CSV output (material inflow & outflow)

    # first, define a function to transpose + combine all variables & add columns to identify material, area & appartment type. Only for csv output
    length = 2
    tag = ['inflow', 'outflow']

    def preprocess(inflow, outflow, area, building, material):
       output_combined = [[]] * length
       output_combined[0] = inflow.transpose()
       output_combined[1] = outflow.transpose()
       for item in range(0,length):
          output_combined[item].insert(0,'material', [material] * 26)
          output_combined[item].insert(0,'area', [area] * 26)
          output_combined[item].insert(0,'type', [building] * 26)
          output_combined[item].insert(0,'flow', [tag[item]] * 26)
       return output_combined

    # cement output
    kg_det_rur_cement_out  = preprocess(kg_det_rur_cement_i,  kg_det_rur_cement_o,  'rural','detached', 'cement') 
    kg_sem_rur_cement_out  = preprocess(kg_sem_rur_cement_i,  kg_sem_rur_cement_o,  'rural','semi-detached', 'cement') 
    kg_app_rur_cement_out  = preprocess(kg_app_rur_cement_i,  kg_app_rur_cement_o,  'rural','appartments', 'cement') 
    kg_hig_rur_cement_out  = preprocess(kg_hig_rur_cement_i,  kg_hig_rur_cement_o,  'rural','high-rise', 'cement') 

    kg_det_urb_cement_out  = preprocess(kg_det_urb_cement_i,  kg_det_urb_cement_o,  'urban','detached', 'cement') 
    kg_sem_urb_cement_out  = preprocess(kg_sem_urb_cement_i,  kg_sem_urb_cement_o,  'urban','semi-detached', 'cement') 
    kg_app_urb_cement_out  = preprocess(kg_app_urb_cement_i,  kg_app_urb_cement_o,  'urban','appartments', 'cement') 
    kg_hig_urb_cement_out  = preprocess(kg_hig_urb_cement_i,  kg_hig_urb_cement_o,  'urban','high-rise', 'cement') 

    kg_office_cement_out  = preprocess(kg_office_cement_i,  kg_office_cement_o,  'commercial','office', 'cement')
    kg_retail_cement_out  = preprocess(kg_retail_cement_i,  kg_retail_cement_o,  'commercial','retail', 'cement')
    kg_hotels_cement_out  = preprocess(kg_hotels_cement_i,  kg_hotels_cement_o,  'commercial','hotels', 'cement')
    kg_govern_cement_out  = preprocess(kg_govern_cement_i,  kg_govern_cement_o,  'commercial','govern', 'cement')


    cement_inflow_total = (
        kg_det_rur_cement_i + kg_sem_rur_cement_i + kg_app_rur_cement_i + kg_hig_rur_cement_i
        + kg_det_urb_cement_i + kg_sem_urb_cement_i + kg_app_urb_cement_i + kg_hig_urb_cement_i
        + kg_office_cement_i + kg_retail_cement_i + kg_hotels_cement_i + kg_govern_cement_i
    )
    cement_inflow_total.name = "cement_inflow_total"
    cement_inflow_path = output_dir / f"cement_demand_inflow_{scenario_code}.csv"
    cement_inflow_total.to_csv(cement_inflow_path)

    os.chdir(old_cwd)
    return {
        "scenario_code": scenario_code,
        "cement_inflow_total": cement_inflow_total,
        "cement_inflow_path": str(cement_inflow_path),
        "cement_output_path": str(cement_inflow_path),
    }
