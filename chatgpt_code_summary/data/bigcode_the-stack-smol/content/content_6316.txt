import wmi
import speedtest_cli
import threading
import signal
import os
import json

def testSpeed(urls):
    speedtest_cli.shutdown_event = threading.Event()
    signal.signal(signal.SIGINT, speedtest_cli.ctrl_c)

    print "Start to test download speed: "
    dlspeed = speedtest_cli.downloadSpeed(urls)
    dlspeed = (dlspeed / 1000 / 1000)
    print('Download: %0.2f M%s/s' % (dlspeed, 'B'))

    return dlspeed

def setGateway(wmiObj, gateway):
    ip = '192.168.8.84'
    subnetmask = '255.255.255.0'
    configurations = wmiObj.Win32_NetworkAdapterConfiguration(Description="Realtek PCIe GBE Family Controller", IPEnabled=True)

    if len(configurations) == 0:
        print "No service available"
        return

    configuration = configurations[0]
    # ret = configuration.EnableStatic(IPAddress=[ip],SubnetMask=[subnetmask])
    ret = configuration.SetGateways(DefaultIPGateway=[gateway])

    return ret

def checkGatewayStatus(urls):
    if not urls:
        urls = ["http://www.dynamsoft.com/assets/images/logo-index-dwt.png", "http://www.dynamsoft.com/assets/images/logo-index-dnt.png", "http://www.dynamsoft.com/assets/images/logo-index-ips.png", "http://www.codepool.biz/wp-content/uploads/2015/06/django_dwt.png", "http://www.codepool.biz/wp-content/uploads/2015/07/drag_element.png"]

    # Query current gateway
    wmiObj = wmi.WMI()
    sql = "select IPAddress,DefaultIPGateway from Win32_NetworkAdapterConfiguration where Description=\"Realtek PCIe GBE Family Controller\" and IPEnabled=TRUE"
    configurations = wmiObj.query(sql)

    currentGateway = None
    for configuration in configurations:
        currentGateway = configuration.DefaultIPGateway[0]
        print "IPAddress:", configuration.IPAddress[0], "DefaultIPGateway:", currentGateway

    dlspeed = testSpeed(urls)
    bestChoice = (currentGateway, dlspeed)
    print "Init choice: " + str(bestChoice)

    gateways = ["192.168.8.1", "192.168.8.2"] # define gateways
    settingReturn = 0
    gateways.remove(currentGateway)

    for gateway in gateways:
        settingReturn = setGateway(wmiObj, gateway)

        if (settingReturn[0] != 0):
            print "Setting failed"
            return

        print "Set gateway: " + gateway
        dlspeed = testSpeed(urls)
        option = (gateway, dlspeed)
        print "Network option: " + str(option)

        if (option[1] > bestChoice[1]):
            bestChoice = option

    print "Best choice: " + str(bestChoice)
    setGateway(wmiObj, bestChoice[0])

    try:
        input("Press any key to continue: ")
    except:
        print('Finished')

def readConfigurationFile():
    urls = None
    config = 'config.json'
    if os.path.exists(config):
        with open(config) as file:
            content = file.read()
            try:
                config_json = json.loads(content)
                urls = config_json['urls']
            except:
                pass

    return urls

def main():
    urls = readConfigurationFile()
    checkGatewayStatus(urls)

if __name__ == '__main__':
    main()
