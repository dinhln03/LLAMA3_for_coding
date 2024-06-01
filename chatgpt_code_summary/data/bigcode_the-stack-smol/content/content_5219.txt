from mininet.topo import Topo


class Project1_Topo_0866007(Topo):
    def __init__(self):
        Topo.__init__(self)

        # Add hosts
        h1 = self.addHost('h1', ip='192.168.0.1/24')
        h2 = self.addHost('h2', ip='192.168.0.2/24')
        h3 = self.addHost('h3', ip='192.168.0.3/24')
        h4 = self.addHost('h4', ip='192.168.0.4/24')

        # Add switches
        s1 = self.addSwitch('s1')
        s2 = self.addSwitch('s2')
        s3 = self.addSwitch('s3')
        s4 = self.addSwitch('s4')

        # Add links
        self.addLink(h1, s1)
        self.addLink(h2, s1)
        self.addLink(h3, s2)
        self.addLink(h4, s2)
        self.addLink(s1, s3)
        self.addLink(s1, s4)
        self.addLink(s2, s3)
        self.addLink(s2, s4)


topos = {'topo_0866007': Project1_Topo_0866007}
