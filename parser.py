'''
Parse a set of excel files in a specfic directory
'''

import sys
import pandas as pd
from os import listdir
from os.path import isfile, join
from openpyxl import load_workbook
if __name__ == "__main__":

    config = {"mode":'w', 'startrow':0, "n_runs":50, "directory": "Total Power", "sheet_name": "", "output_name": "results.xlsx"}


    cmd_args = sys.argv[1:]

    for arg in cmd_args:
        parameter_and_input = arg.split("=")
        parameter = parameter_and_input[0]
        inp = parameter_and_input[1]
        config[parameter] = inp

    if len(cmd_args) > 0:
        directory = cmd_args[0]
        if len(cmd_args)>1:
            n_runs = cmd_args[1]
            if len(cmd_args)>2:
                if cmd_args[2] > 0:
                    mode = 'a'

                if len(cmd_args)>3:
                    startrow = int(cmd_args[3])
    if config["sheet_name"] == "":
        config["sheet_name"] = "seeded_" + str(config["n_runs"]) + "runs_clustering"
    print("parsing with the following configurations")
    # default options
    mode = config["mode"]
    startrow = int(config["startrow"])
    n_runs = int(config["n_runs"])
    directory = config["directory"]
    sheet_name = config["sheet_name"]
    output_name = config["output_name"]

    if mode == "a":
        writer = pd.ExcelWriter(output_name, mode=mode, engine="openpyxl", if_sheet_exists="overlay")
    else:
        writer = pd.ExcelWriter(output_name, mode=mode, engine="openpyxl")
    files = [directory + "/" + f for f in listdir(directory) if isfile(join(directory, f)) and f.split(".")[1]=="xlsx"]

    for file in files:

        estimator = file.split("_")[0].split("/")[1]
        skip = False
        try:

            result = pd.read_excel(file, engine="openpyxl", header=None, sheet_name=sheet_name)
        except ValueError:
            skip = True

        if not skip:
            try:

                row_number = result[result[0] == estimator].index.tolist()[0]
                final_result = result.iloc[row_number:row_number + 1]
                final_result = final_result.drop([0], axis=1)
                final_result.columns = result.iloc[row_number - 1][1:]
                final_result.index = [estimator]
                final_result.to_excel(writer, startrow=startrow, sheet_name=sheet_name)
                startrow = startrow + 3
            except IndexError:
                skip = True
        if skip:
            print("the file %s was skipped" %file)
        else:
            print("the file %s was read successfully" %file)


    writer.save()
    print("\nParsing is Done!")
    print("parsed results are saved as %s" %(output_name))