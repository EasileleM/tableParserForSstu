import pygsheets
from string import Template

AMOUNT_OF_VERTICES = 27
AMOUNT_OF_FACS = 6
# init functions here
FUNCTIONS = [lambda x: 1] * AMOUNT_OF_VERTICES ** 2
FUNCTIONS[0] = lambda x: x / 2
FUNCTIONS[1] = lambda x: x ** 2 + 2 * x
M_ASTERISK_ARRAY = [1] * AMOUNT_OF_VERTICES

gc = pygsheets.authorize(service_file='service.json')

# Open spreadsheet and then worksheet
sheet1 = gc.open('tableKush').sheet1

matrixOfVertices = sheet1.get_values((2, 2), (AMOUNT_OF_VERTICES + 1, AMOUNT_OF_VERTICES + 1))

matrixOfFacs = sheet1.get_values((2, AMOUNT_OF_VERTICES + 2),
                                 (AMOUNT_OF_VERTICES + AMOUNT_OF_FACS + 1, AMOUNT_OF_VERTICES + AMOUNT_OF_FACS + 1))

formulas = []
calculations = []


def createVerticesFormulaPart(verticesStr, fNumber, verticeNumber):
    return (' * ' if len(verticesStr) else '') + 'f' + str(fNumber) + '(M' + str(verticeNumber) + '(t))'


def createFacFormulaPart(facStr, facNumber):
    return (' + ' if len(facStr) else '') + 'Fac' + str(facNumber) + '(t)'


fNumber = 1
for y in range(AMOUNT_OF_VERTICES):

    # B and D formulas with only vertices parts
    BV = ''
    BV_VALUE = 1
    DV = ''
    DV_VALUE = 1

    for x in range(AMOUNT_OF_VERTICES):
        if matrixOfVertices[y][x] == '1':
            BV += createVerticesFormulaPart(BV, fNumber, x + 1)

            # TODO how to calculate properly? What is M1(t)?
            BV_VALUE *= FUNCTIONS[fNumber - 1](1)  # f(M(t))
            fNumber += 1
        elif matrixOfVertices[y][x] == '-1':
            DV += createVerticesFormulaPart(DV, fNumber, x + 1)

            # TODO how to calculate properly? What is M1(t)?
            DV_VALUE *= FUNCTIONS[fNumber - 1](1)  # f(M(t))
            fNumber += 1



    # B and D formulas with only facs parts
    BF = ''
    BF_VALUE = 0
    DF = ''
    DF_VALUE = 0

    for x in range(AMOUNT_OF_FACS):
        if matrixOfFacs[y][x] == '1':
            BF += createFacFormulaPart(BF, x + 1)

            # TODO how to calculate properly? What is Fac1(t)?
            BF_VALUE += 1  # Fac(t)
        elif matrixOfFacs[y][x] == '-1':
            DF += createFacFormulaPart(DF, x + 1)

            # TODO how to calculate properly? What is Fac1(t)?
            DF_VALUE += 1  # Fac(t)

    BDelimiter = ' * ' if len(BV) and len(BF) else ''
    DDelimiter = ' * ' if len(DV) and len(DF) else ''

    B = BV + BDelimiter + ('(' + BF + ')' if len(BF) else '')
    D = DV + DDelimiter + ('(' + DF + ')' if len(DF) else '')

    formula = 'dM' + str(y + 1) + '(t) / dt = (' + B + ' - ' + D + ')/M' + str(y + 1) + '*'

    # TODO how to calculate properly? What is M1*?
    calculation = (BV_VALUE * BF_VALUE - DV_VALUE * DF_VALUE) / M_ASTERISK_ARRAY[y]

    formulas.append(formula)
    calculations.append(calculation)

# printing and persisting

TITLE = 'FORMULAS AND CALCULATIONS\n\n'
DELIMITER_FORMULA_MSG_TEMPLATE = Template(
    '\n\n-----------------------------FORMULA $num-----------------------------------\n')
DELIMITER_CALCULATION_MSG_TEMPLATE = Template(
    '\n-----------------------------CALCULATION $num-----------------------------------\n\n')

with open('results.txt', 'w') as file:
    print(TITLE)
    file.write(TITLE)
    for index in range(AMOUNT_OF_VERTICES):
        print(DELIMITER_FORMULA_MSG_TEMPLATE.substitute(num=index + 1))
        file.write(DELIMITER_FORMULA_MSG_TEMPLATE.substitute(num=index + 1))
        print(formulas[index])
        file.write(formulas[index])
        print(DELIMITER_CALCULATION_MSG_TEMPLATE.substitute(num=index + 1))
        file.write(DELIMITER_CALCULATION_MSG_TEMPLATE.substitute(num=index + 1))
        print(calculations[index])
        file.write(str(calculations[index]))

    file.close()
