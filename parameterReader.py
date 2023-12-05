import sys

"""
    Class to simplify reading parameters
    Given a dictionary of expected default parameters, it will parse a list of possible parameters
    Will throw NameError on incorrect usage of parameters / unknown parameter
"""
class ParameterReader:
    """
        Input:
            params:dict = Default set of parameters (do not set any to None)
    """
    def __init__(self, params:dict):
        self.params = params

    """
        Processes a list of parameters, applying values to known parameters
        Input:
            ar:list = List of strings representing parameters

        Exceptions:
            NameError on any invalid parameter name/setting
    """
    def setParams(self, ar):
        param = None
        for item in ar[1:]:
            if param == None:
                if self.params.get(item) != None:
                    if item[0:2] == "--":
                        self.params[item] = True
                    elif item[0] == '-':
                        param = item
                    else:
                        raise NameError(f"Poor format: {item}")
                else:
                    raise NameError(f"No setting for: {item}")
            else:
                if item[0] == '-':
                    raise NameError(f"Cannot set setting {param} with setting: {item}")
                else:
                    self.params[param] = item
                    param = None
        
        if param != None:
            raise NameError(f"No data item for: {param}")
                
    """
        Returns newly set values as the original set (Call after setParams()) (Getter method)
        Return:
            dict
    """
    def getParams(self):
        return self.params


"""
    Testing for ParameterReader 
"""
def test():
    params = {"--save": False, "--load": False, "-file": ""}
    pr = ParameterReader(params)

    try:
        pr.setParams(sys.argv)
    except NameError as e:
        print(f"Error: {e.args[0]}")
    
    print(pr.getParams())


if __name__=="__main__":
    test()