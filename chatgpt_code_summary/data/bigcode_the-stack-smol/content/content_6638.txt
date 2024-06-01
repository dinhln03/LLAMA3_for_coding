import collections
import itertools
import string
import unittest

# noinspection PyUnusedLocal
# skus = unicode string
def getItemPrices():
    itemPrices = {}
    itemPrices['A'] = {1:50, 3:130, 5:200}
    itemPrices['B'] = {1:30, 2:45}
    itemPrices['C'] = {1:20}
    itemPrices['D'] = {1:15}
    itemPrices['E'] = {1:40}
    itemPrices['F'] = {1:10}
    itemPrices['G'] = {1:20}
    itemPrices['H'] = {1:10, 5:45, 10:80}
    itemPrices['I'] = {1:35}
    itemPrices['J'] = {1:60}
    itemPrices['K'] = {1:70, 2:120}
    itemPrices['L'] = {1:90}
    itemPrices['M'] = {1:15}
    itemPrices['N'] = {1:40}
    itemPrices['O'] = {1:10}
    itemPrices['P'] = {1:50, 5:200}
    itemPrices['Q'] = {1:30, 3:80}
    itemPrices['R'] = {1:50}
    itemPrices['S'] = {1:20}
    itemPrices['T'] = {1:20}
    itemPrices['U'] = {1:40}
    itemPrices['V'] = {1:50, 2:90, 3:130}
    itemPrices['W'] = {1:20}
    itemPrices['X'] = {1:17}
    itemPrices['Y'] = {1:20}
    itemPrices['Z'] = {1:21}

    return itemPrices

def getGroupItemPrices():
    itemPrices = getItemPrices()

    groupPrices = {}
    for combination in itertools.combinations_with_replacement("STXYZ", 3):
        regularCost = sum(itemPrices[item][1] for item in combination)
        saving = regularCost - 45
        # FIXME: Using 0 to denote saving from using group
        groupPrices["".join(combination)] = {1:45, 0:saving}

    return groupPrices

def getItemFreebies():
    itemFreebies = {}
    itemFreebies['E'] = {2:'B'}
    itemFreebies['F'] = {3:'F'}
    itemFreebies['N'] = {3:'M'}
    itemFreebies['R'] = {3:'Q'}
    itemFreebies['U'] = {4:'U'}

    return itemFreebies

def generateItemCounts(skus):
    itemCounts = collections.defaultdict(int)
    for item in skus:

        invalidItem = item not in string.ascii_uppercase
        if invalidItem:
            raise ValueError
        else:
            itemCounts[item] += 1

    return itemCounts

def removeFreeItems(itemCounts):
    itemFreebies = getItemFreebies()

    freeItems = {}
    for item, count in itemCounts.items():
        freebies = itemFreebies.get(item, {})

        for itemsNeededForFreebe, freeItem in freebies.items():
            freebeeCount = int(count/itemsNeededForFreebe)
            freeItems[freeItem] = freebeeCount

    for freeItem, count in freeItems.items():
        itemCounts[freeItem] = max(0, itemCounts[freeItem] - count)

def applyItemGroupings(itemCounts):
    groupItemPrices = getGroupItemPrices()
    groupsByLargestSaving = sorted(list(groupItemPrices.keys()), key = lambda group: groupItemPrices[group][0], reverse=True)

    for group in groupsByLargestSaving:

        while True:
            groupCounts = collections.defaultdict(int)

            for groupItem in group:
                if itemCounts[groupItem]:
                    groupCounts[groupItem] += 1
                    itemCounts[groupItem] -= 1
                else:
                    for item, count in groupCounts.items():
                        itemCounts[item] += count
                    break
            else:
                itemCounts[group] += 1
                continue

            break

def calculateItemCosts(itemCounts):
    itemPrices = getItemPrices()
    itemPrices.update(getGroupItemPrices())

    totalCost = 0
    for item, count in itemCounts.items():

        prices = itemPrices[item]
        for n in reversed(list(prices.keys())):
            if n == 0:
                continue

            price = prices[n]

            offerCount = int(count/n)
            totalCost += offerCount * price
            count -= offerCount * n

    return totalCost

def checkout(skus):
    try:
        itemCounts = generateItemCounts(skus)
    except ValueError:
        return -1

    removeFreeItems(itemCounts)
    applyItemGroupings(itemCounts)

    return calculateItemCosts(itemCounts)

class TestCheckOut(unittest.TestCase):
    def test_invalidSKUItemReturnsMinus1(self):
        self.assertEqual(checkout("AB32"), -1)
        self.assertEqual(checkout("ABc"), -1)
        self.assertEqual(checkout("AB!"), -1)

    def test_emptySKUCostsNothing(self):
        self.assertEqual(checkout(""), 0)

    def test_singlePrices(self):
        self.assertEqual(checkout('A'), 50)
        self.assertEqual(checkout('B'), 30)
        self.assertEqual(checkout('C'), 20)
        self.assertEqual(checkout('D'), 15)
        self.assertEqual(checkout('E'), 40)
        self.assertEqual(checkout('F'), 10)
        self.assertEqual(checkout('G'), 20)
        self.assertEqual(checkout('H'), 10)
        self.assertEqual(checkout('I'), 35)
        self.assertEqual(checkout('J'), 60)
        self.assertEqual(checkout('K'), 70)
        self.assertEqual(checkout('L'), 90)
        self.assertEqual(checkout('M'), 15)
        self.assertEqual(checkout('N'), 40)
        self.assertEqual(checkout('O'), 10)
        self.assertEqual(checkout('P'), 50)
        self.assertEqual(checkout('Q'), 30)
        self.assertEqual(checkout('R'), 50)
        self.assertEqual(checkout('S'), 20)
        self.assertEqual(checkout('T'), 20)
        self.assertEqual(checkout('U'), 40)
        self.assertEqual(checkout('V'), 50)
        self.assertEqual(checkout('W'), 20)
        self.assertEqual(checkout('X'), 17)
        self.assertEqual(checkout('Y'), 20)
        self.assertEqual(checkout('Z'), 21)

    def test_multipleItemOffers(self):
        self.assertEqual(checkout('AAA'), 130)
        self.assertEqual(checkout('AAAAA'), 200)
        self.assertEqual(checkout('BB'), 45)
        self.assertEqual(checkout("HHHHH"), 45)
        self.assertEqual(checkout("HHHHHHHHHH"), 80)
        self.assertEqual(checkout("KK"), 120)
        self.assertEqual(checkout("PPPPP"), 200)
        self.assertEqual(checkout("QQQ"), 80)
        self.assertEqual(checkout("VV"), 90)
        self.assertEqual(checkout("VVV"), 130)

    def test_multipleNonOfferItemsAreMultiplesOfSingleItemPrice(self):
        self.assertEqual(checkout('CC'), checkout('C') * 2)
        self.assertEqual(checkout('DD'), checkout('D') * 2)

    def test_mixedSingleItemsAreSumOfIndividualPrices(self):
        self.assertEqual(checkout("BADC"), checkout("A") + checkout("B") + checkout("C") + checkout("D"))

    def test_multipleSpecialOffserAreMultipleOfSpecialOfferPrice(self):
        self.assertEqual(checkout("AAAAAAAAAA"), checkout("AAAAA") * 2)
        self.assertEqual(checkout("BBBB"), checkout("BB") * 2)

    def test_mixedOffersAreSumOfSpecialAndIndividualPrices(self):
        self.assertEqual(checkout("AAAAAAA"), checkout("AAAAA") + checkout("AA"))
        self.assertEqual(checkout("BBB"), checkout("BB") + checkout("B"))

    def test_mixedSpecialOffersAreSumsOfOffers(self):
        self.assertEqual(checkout("ABABA"), checkout("BB") + checkout("AAA"))

    def test_mixedItemsAreSumed(self):
        self.assertEqual(checkout("ABCCABADDA"), checkout("BB") + checkout("AAA") + checkout("A") + checkout("CC") + checkout("DD"))

    def test_specialOfferCombinationsMinimisePrice(self):
        self.assertEqual(checkout("AAAAAAAAA"), checkout("AAAAA") + checkout("AAA") + checkout("A"))

    def test_2ESpecialOfferGivesOneFreeB(self):
        self.assertEqual(checkout("EE"), checkout("E") + checkout("E"))
        self.assertEqual(checkout("EEB"), checkout("E") + checkout("E"))
        self.assertEqual(checkout("EEBEE"), checkout("E") * 4)
        self.assertEqual(checkout("EEBEEB"), checkout("E") * 4)
        self.assertEqual(checkout("EEBEEBB"), checkout("E") * 4 + checkout("B"))

    def test_3FSpecialOfferGivesOneFreeF(self):
        self.assertEqual(checkout("FFF"), checkout("F") * 2)
        self.assertEqual(checkout("FFFFF"), checkout("F") * 4)
        self.assertEqual(checkout("FFFFFF"), checkout("F") * 4)

    def test_3NSpecialOfferGivesOneFreeM(self):
        self.assertEqual(checkout("NNNM"), checkout("NNN"))

    def test_3RSpecialOfferGivesOneFreeQ(self):
        self.assertEqual(checkout("RRRQ"), checkout("RRR"))

    def test_4USpecialOfferGivesOneFreeU(self):
        self.assertEqual(checkout("UUUU"), checkout("UUU"))

    def test_groupDiscount(self):
        for combination in itertools.combinations_with_replacement("STXYZ", 3):
            self.assertEqual(checkout("".join(combination)), 45)

    def test_maximumGroupDiscount(self):
        self.assertEqual(checkout("STXYZ"), 45 + checkout("XY"))
        self.assertEqual(checkout("SSSX"), 45 + checkout("X"))

    def test_multipleGroupDiscountsAreGiven(self):
        self.assertEqual(checkout("STXYZTYX"), 90 + checkout("XX"))




if __name__ == '__main__':
    unittest.main()






