from lib.operation_definition import NodeOperation
from utils.common import create_directory, get_parent_dir

def csv_writer(node):
    writer = NodeOperation(node) # initiate
    writer.io_check(len(writer.inputs) == len(writer.outputs)) # check if i/o expectations are met
    df_list, output_loc_list = writer.get_dataframe_objects_to_write()

    for i in range(0,len(df_list)):
        create_directory(get_parent_dir(output_loc_list[i]))
        df_list[i].to_csv(output_loc_list[i],**writer.arguments) # write to file