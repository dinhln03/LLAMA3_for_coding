import logging, ast, os
from bisect import bisect_left, bisect
import louie as dispatcher
from twisted.internet import reactor
from rdflib import Literal
from light9 import showconfig
from light9.namespaces import L9, RDF, RDFS
from rdfdb.patch import Patch

log = logging.getLogger()
# todo: move to config, consolidate with ascoltami, musicPad, etc
introPad = 4
postPad = 4


class Curve(object):
    """curve does not know its name. see Curveset"""

    def __init__(self, uri, pointsStorage='graph'):
        self.uri = uri
        self.pointsStorage = pointsStorage
        self.points = []  # x-sorted list of (x,y)
        self._muted = False

    def __repr__(self):
        return "<%s %s (%s points)>" % (self.__class__.__name__, self.uri,
                                        len(self.points))

    def muted():
        doc = "Whether to currently send levels (boolean, obviously)"

        def fget(self):
            return self._muted

        def fset(self, val):
            self._muted = val
            dispatcher.send('mute changed', sender=self)

        return locals()

    muted = property(**muted())

    def toggleMute(self):
        self.muted = not self.muted

    def load(self, filename):
        self.points[:] = []
        for line in open(filename):
            x, y = line.split()
            self.points.append((float(x), ast.literal_eval(y)))
        self.points.sort()
        dispatcher.send("points changed", sender=self)

    def set_from_string(self, pts):
        self.points[:] = []
        vals = pts.split()
        pairs = list(zip(vals[0::2], vals[1::2]))
        for x, y in pairs:
            self.points.append((float(x), ast.literal_eval(y)))
        self.points.sort()
        dispatcher.send("points changed", sender=self)

    def points_as_string(self):

        def outVal(x):
            if isinstance(x, str):  # markers
                return x
            return "%.4g" % x

        return ' '.join(
            "%s %s" % (outVal(p[0]), outVal(p[1])) for p in self.points)

    def save(self, filename):
        # this is just around for markers, now
        if filename.endswith('-music') or filename.endswith('_music'):
            print("not saving music track")
            return
        f = open(filename, 'w')
        for p in self.points:
            f.write("%s %r\n" % p)
        f.close()

    def eval(self, t, allow_muting=True):
        if self.muted and allow_muting:
            return 0
        if not self.points:
            raise ValueError("curve has no points")
        i = bisect_left(self.points, (t, None)) - 1

        if i == -1:
            return self.points[0][1]
        if self.points[i][0] > t:
            return self.points[i][1]
        if i >= len(self.points) - 1:
            return self.points[i][1]

        p1, p2 = self.points[i], self.points[i + 1]
        frac = (t - p1[0]) / (p2[0] - p1[0])
        y = p1[1] + (p2[1] - p1[1]) * frac
        return y

    __call__ = eval

    def insert_pt(self, new_pt):
        """returns index of new point"""
        i = bisect(self.points, (new_pt[0], None))
        self.points.insert(i, new_pt)
        # missing a check that this isn't the same X as the neighbor point
        dispatcher.send("points changed", sender=self)
        return i

    def live_input_point(self, new_pt, clear_ahead_secs=.01):
        x, y = new_pt
        exist = self.points_between(x, x + clear_ahead_secs)
        for pt in exist:
            self.remove_point(pt)
        self.insert_pt(new_pt)
        dispatcher.send("points changed", sender=self)
        # now simplify to the left

    def set_points(self, updates):
        for i, pt in updates:
            self.points[i] = pt

        # this should be on, but live_input_point made it fail a
        # lot. need a new solution.
        #self.checkOverlap()
        dispatcher.send("points changed", sender=self)

    def checkOverlap(self):
        x = None
        for p in self.points:
            if p[0] <= x:
                raise ValueError("overlapping points")
            x = p[0]

    def pop_point(self, i):
        p = self.points.pop(i)
        dispatcher.send("points changed", sender=self)
        return p

    def remove_point(self, pt):
        self.points.remove(pt)
        dispatcher.send("points changed", sender=self)

    def indices_between(self, x1, x2, beyond=0):
        leftidx = max(0, bisect(self.points, (x1, None)) - beyond)
        rightidx = min(len(self.points),
                       bisect(self.points, (x2, None)) + beyond)
        return list(range(leftidx, rightidx))

    def points_between(self, x1, x2):
        """returns (x,y) points"""
        return [self.points[i] for i in self.indices_between(x1, x2)]

    def point_before(self, x):
        """(x,y) of the point left of x, or None"""
        leftidx = self.index_before(x)
        if leftidx is None:
            return None
        return self.points[leftidx]

    def index_before(self, x):
        leftidx = bisect(self.points, (x, None)) - 1
        if leftidx < 0:
            return None
        return leftidx


class CurveResource(object):
    """
    holds a Curve, deals with graphs
    """

    def __init__(self, graph, uri):
        # probably newCurve and loadCurve should be the constructors instead.
        self.graph, self.uri = graph, uri

    def curvePointsContext(self):
        return self.uri

    def newCurve(self, ctx, label):
        """
        Save type/label for a new :Curve resource.
        Pass the ctx where the main curve data (not the points) will go.
        """
        if hasattr(self, 'curve'):
            raise ValueError('CurveResource already has a curve %r' %
                             self.curve)
        self.graph.patch(
            Patch(addQuads=[
                (self.uri, RDF.type, L9['Curve'], ctx),
                (self.uri, RDFS.label, label, ctx),
            ]))
        self.curve = Curve(self.uri)
        self.curve.points.extend([(0, 0)])
        self.saveCurve()
        self.watchCurvePointChanges()

    def loadCurve(self):
        if hasattr(self, 'curve'):
            raise ValueError('CurveResource already has a curve %r' %
                             self.curve)
        pointsFile = self.graph.value(self.uri, L9['pointsFile'])
        self.curve = Curve(self.uri,
                           pointsStorage='file' if pointsFile else 'graph')
        if hasattr(self.graph, 'addHandler'):
            self.graph.addHandler(self.pointsFromGraph)
        else:
            # given a currentState graph
            self.pointsFromGraph()

    def pointsFromGraph(self):
        pts = self.graph.value(self.uri, L9['points'])
        if pts is not None:
            self.curve.set_from_string(pts)
        else:
            diskPts = self.graph.value(self.uri, L9['pointsFile'])
            if diskPts is not None:
                self.curve.load(os.path.join(showconfig.curvesDir(), diskPts))
            else:
                log.warn("curve %s has no points", self.uri)
        self.watchCurvePointChanges()

    def saveCurve(self):
        self.pendingSave = None
        for p in self.getSavePatches():
            self.graph.patch(p)

    def getSavePatches(self):
        if self.curve.pointsStorage == 'file':
            log.warn("not saving file curves anymore- skipping %s" % self.uri)
            #cur.save("%s-%s" % (basename,name))
            return []
        elif self.curve.pointsStorage == 'graph':
            return [
                self.graph.getObjectPatch(self.curvePointsContext(),
                                          subject=self.uri,
                                          predicate=L9['points'],
                                          newObject=Literal(
                                              self.curve.points_as_string()))
            ]
        else:
            raise NotImplementedError(self.curve.pointsStorage)

    def watchCurvePointChanges(self):
        """start watching and saving changes to the graph"""
        dispatcher.connect(self.onChange, 'points changed', sender=self.curve)

    def onChange(self):

        # Don't write a patch for the edited curve points until they've been
        # stable for this long. This can be very short, since it's just to
        # stop a 100-point edit from sending many updates. If it's too long,
        # you won't see output lights change while you drag a point.  Todo:
        # this is just the wrong timing algorithm- it should be a max rate,
        # not a max-hold-still-time.
        HOLD_POINTS_GRAPH_COMMIT_SECS = .1

        if getattr(self, 'pendingSave', None):
            self.pendingSave.cancel()
        self.pendingSave = reactor.callLater(HOLD_POINTS_GRAPH_COMMIT_SECS,
                                             self.saveCurve)


class Markers(Curve):
    """Marker is like a point but the y value is a string"""

    def eval(self):
        raise NotImplementedError()


def slope(p1, p2):
    if p2[0] == p1[0]:
        return 0
    return (p2[1] - p1[1]) / (p2[0] - p1[0])


class Curveset(object):

    def __init__(self, graph, session):
        self.graph, self.session = graph, session

        self.currentSong = None
        self.curveResources = {}  # uri : CurveResource

        self.markers = Markers(uri=None, pointsStorage='file')

        graph.addHandler(self.loadCurvesForSong)

    def curveFromUri(self, uri):
        return self.curveResources[uri].curve

    def loadCurvesForSong(self):
        """
        current curves will track song's curves.
        
        This fires 'add_curve' dispatcher events to announce the new curves.
        """
        log.info('loadCurvesForSong')
        dispatcher.send("clear_curves")
        self.curveResources.clear()
        self.markers = Markers(uri=None, pointsStorage='file')

        self.currentSong = self.graph.value(self.session, L9['currentSong'])
        if self.currentSong is None:
            return

        for uri in sorted(self.graph.objects(self.currentSong, L9['curve'])):
            try:
                cr = self.curveResources[uri] = CurveResource(self.graph, uri)
                cr.loadCurve()

                curvename = self.graph.label(uri)
                if not curvename:
                    raise ValueError("curve %r has no label" % uri)
                dispatcher.send("add_curve",
                                sender=self,
                                uri=uri,
                                label=curvename,
                                curve=cr.curve)
            except Exception as e:
                log.error("loading %s failed: %s", uri, e)

        basename = os.path.join(
            showconfig.curvesDir(),
            showconfig.songFilenameFromURI(self.currentSong))
        try:
            self.markers.load("%s.markers" % basename)
        except IOError:
            print("no marker file found")

    def save(self):
        """writes a file for each curve with a name
        like basename-curvename, or saves them to the rdf graph"""
        basename = os.path.join(
            showconfig.curvesDir(),
            showconfig.songFilenameFromURI(self.currentSong))

        patches = []
        for cr in list(self.curveResources.values()):
            patches.extend(cr.getSavePatches())

        self.markers.save("%s.markers" % basename)
        # this will cause reloads that will rebuild our curve list
        for p in patches:
            self.graph.patch(p)

    def sorter(self, name):
        return self.curves[name].uri

    def curveUrisInOrder(self):
        return sorted(self.curveResources.keys())

    def currentCurves(self):
        # deprecated
        for uri, cr in sorted(self.curveResources.items()):
            with self.graph.currentState(tripleFilter=(uri, RDFS['label'],
                                                       None)) as g:
                yield uri, g.label(uri), cr.curve

    def globalsdict(self):
        raise NotImplementedError('subterm used to get a dict of name:curve')

    def get_time_range(self):
        return 0, dispatcher.send("get max time")[0][1]

    def new_curve(self, name):
        if isinstance(name, Literal):
            name = str(name)

        uri = self.graph.sequentialUri(self.currentSong + '/curve-')

        cr = self.curveResources[uri] = CurveResource(self.graph, uri)
        cr.newCurve(ctx=self.currentSong, label=Literal(name))
        s, e = self.get_time_range()
        cr.curve.points.extend([(s, 0), (e, 0)])

        ctx = self.currentSong
        self.graph.patch(
            Patch(addQuads=[
                (self.currentSong, L9['curve'], uri, ctx),
            ]))
        cr.saveCurve()
