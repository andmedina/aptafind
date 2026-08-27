'''
This script will generate a full sequence when inputing the left flanking region, the core region, and the right flanking region.
'''

def collectCore():
    return input("Please enter the core sequence: ")

def collectFlankingRegions():
    left_flank = input("Please enter the left flanking region: ")
    right_flank = input("Please enter the right flanking region: ")
    return left_flank, right_flank

def computeFullSequence(left, core, right):
    return left + core + right


ans = 1
core = ""
flanking = ""
while (ans != 0):   

    ans = int(input("What would you like to do: \n0. Exit\n1.Enter new core sequence\n2.Enter new flanking regions\n3.Compute sequence"))

    if ans == 0:
        exit
    elif ans == 1:
        core = collectCore()
    elif ans == 2:
        flanking = collectFlankingRegions()
    elif ans == 3:
        print(computeFullSequence(flanking[0], core, flanking[1]))
    else:
        "Please enter a valid response."

    







