# encoding: utf-8
# module Autodesk.Civil.Settings calls itself Settings
# from AeccDbMgd, Version=13.3.854.0, Culture=neutral, PublicKeyToken=null, AeccPressurePipesMgd, Version=13.3.854.0, Culture=neutral, PublicKeyToken=null
# by generator 1.145
# no doc
# no imports

# no functions
# classes

class AbbreviationAlignmentEnhancedType(Enum):
    """ enum AbbreviationAlignmentEnhancedType, values: AlignmentBeginningPoint (402706556), AlignmentEndPoint (402706557), CompoundSpiralLargeRadiusAtBeginning (402706566), CompoundSpiralLargeRadiusAtEnd (402706567), CompoundSpiralSmallRadiusAtBeginning (402706568), CompoundSpiralSmallRadiusAtEnd (402706569), CurveBeginning (402706560), CurveEnd (402706561), LineBeginning (402706558), LineEnd (402706559), SimpleSpiralLargeRadiusAtBeginning (402706562), SimpleSpiralLargeRadiusAtEnd (402706563), SimpleSpiralSmallRadiusAtBeginning (402706564), SimpleSpiralSmallRadiusAtEnd (402706565) """
    AlignmentBeginningPoint = None
    AlignmentEndPoint = None
    CompoundSpiralLargeRadiusAtBeginning = None
    CompoundSpiralLargeRadiusAtEnd = None
    CompoundSpiralSmallRadiusAtBeginning = None
    CompoundSpiralSmallRadiusAtEnd = None
    CurveBeginning = None
    CurveEnd = None
    LineBeginning = None
    LineEnd = None
    SimpleSpiralLargeRadiusAtBeginning = None
    SimpleSpiralLargeRadiusAtEnd = None
    SimpleSpiralSmallRadiusAtBeginning = None
    SimpleSpiralSmallRadiusAtEnd = None
    value__ = None


class AbbreviationAlignmentType(Enum):
    """ enum AbbreviationAlignmentType, values: AlignmentBeginning (67162235), AlignmentEnd (67162234), CompoundCurveCurveIntersect (67162197), CurveSpiralIntersect (67162201), CurveTangentIntersect (67162196), MidCurvePoint (67162254), ReverseCurveCurveIntersect (67162198), ReverseSpiralIntersect (67162204), SpiralCurveIntersect (67162202), SpiralSpiralIntersect (67162203), SpiralTangentIntersect (67162200), StationEquationDecreasing (67162253), StationEquationIncreasing (67162252), TangentCurveIntersect (67162195), TangentSpiralIntersect (67162199), TangentTangentIntersect (67162194) """
    AlignmentBeginning = None
    AlignmentEnd = None
    CompoundCurveCurveIntersect = None
    CurveSpiralIntersect = None
    CurveTangentIntersect = None
    MidCurvePoint = None
    ReverseCurveCurveIntersect = None
    ReverseSpiralIntersect = None
    SpiralCurveIntersect = None
    SpiralSpiralIntersect = None
    SpiralTangentIntersect = None
    StationEquationDecreasing = None
    StationEquationIncreasing = None
    TangentCurveIntersect = None
    TangentSpiralIntersect = None
    TangentTangentIntersect = None
    value__ = None


class AbbreviationCantType(Enum):
    """ enum AbbreviationCantType, values: BeginAlignment (67163513), BeginFullCant (67163510), BeginLevelRail (67163509), EndAlignment (67163514), EndFullCant (67163511), EndLevelRail (67163508), Manual (67163512) """
    BeginAlignment = None
    BeginFullCant = None
    BeginLevelRail = None
    EndAlignment = None
    EndFullCant = None
    EndLevelRail = None
    Manual = None
    value__ = None


class AbbreviationProfileType(Enum):
    """ enum AbbreviationProfileType, values: BeginVerticalCurve (67173890), BeginVerticalCurveElevation (67173892), BeginVerticalCurveStation (67173891), CurveCoefficient (67173898), EndVerticalCurve (67173893), EndVerticalCurveElevation (67173895), EndVerticalCurveStation (67173894), GradeBreak (67173889), GradeChange (67173899), HighPoint (67173896), LowPoint (67173897), OverallHighPoint (67173909), OverallLowPoint (67173910), PointOfVerticalIntersection (67173888), ProfileEnd (67173902), ProfileStart (67173901), VerticalCompoundCurveIntersect (67173903), VerticalCompoundCurveIntersectElevation (67173906), VerticalCompoundCurveIntersectStation (67173905), VerticalReverseCurveIntersect (67173904), VerticalReverseCurveIntersectElevation (67173908), VerticalReverseCurveIntersectStation (67173907) """
    BeginVerticalCurve = None
    BeginVerticalCurveElevation = None
    BeginVerticalCurveStation = None
    CurveCoefficient = None
    EndVerticalCurve = None
    EndVerticalCurveElevation = None
    EndVerticalCurveStation = None
    GradeBreak = None
    GradeChange = None
    HighPoint = None
    LowPoint = None
    OverallHighPoint = None
    OverallLowPoint = None
    PointOfVerticalIntersection = None
    ProfileEnd = None
    ProfileStart = None
    value__ = None
    VerticalCompoundCurveIntersect = None
    VerticalCompoundCurveIntersectElevation = None
    VerticalCompoundCurveIntersectStation = None
    VerticalReverseCurveIntersect = None
    VerticalReverseCurveIntersectElevation = None
    VerticalReverseCurveIntersectStation = None


class AbbreviationSuperelevationType(Enum):
    """ enum AbbreviationSuperelevationType, values: BeginFullSuper (67163478), BeginNormalCrown (67163476), BeginNormalShoulder (67163480), BeginOfAlignment (67163474), BeginShoulderRollover (67163506), EndFullSuper (67163479), EndNormalCrown (67163477), EndNormalShoulder (67163481), EndOfAlignment (67163475), EndShoulderRollover (67163507), LevelCrown (67163482), LowShoulderMatch (67163483), Manual (67163486), ReverseCrown (67163484), ShoulderBreakover (67163485) """
    BeginFullSuper = None
    BeginNormalCrown = None
    BeginNormalShoulder = None
    BeginOfAlignment = None
    BeginShoulderRollover = None
    EndFullSuper = None
    EndNormalCrown = None
    EndNormalShoulder = None
    EndOfAlignment = None
    EndShoulderRollover = None
    LevelCrown = None
    LowShoulderMatch = None
    Manual = None
    ReverseCrown = None
    ShoulderBreakover = None
    value__ = None


class AutomaticManual(Enum):
    """ enum AutomaticManual, values: Automatic (0), AutomaticObject (1), Manual (2), None (3) """
    Automatic = None
    AutomaticObject = None
    Manual = None
    None = None
    value__ = None


class DrawingUnitType(Enum):
    """ enum DrawingUnitType, values: Feet (30), Meters (2) """
    Feet = None
    Meters = None
    value__ = None


class GeographicCoordinateType(Enum):
    """ enum GeographicCoordinateType, values: LatLong (0), LongLat (1) """
    LatLong = None
    LongLat = None
    value__ = None


class GridCoordinateType(Enum):
    """ enum GridCoordinateType, values: EastingNorthing (0), NorthingEasting (1) """
    EastingNorthing = None
    NorthingEasting = None
    value__ = None


class GridScaleFactorType(Enum):
    """ enum GridScaleFactorType, values: PrismodialFormula (3), ReferencePoint (2), Unity (0), UserDefined (1) """
    PrismodialFormula = None
    ReferencePoint = None
    Unity = None
    UserDefined = None
    value__ = None


class ImperialToMetricConversionType(Enum):
    """ enum ImperialToMetricConversionType, values: InternationalFoot (536870912), UsSurveyFoot (1073741824) """
    InternationalFoot = None
    UsSurveyFoot = None
    value__ = None


class LandXMLAngularUnits(Enum):
    """ enum LandXMLAngularUnits, values: DegreesDecimal (0), DegreesDms (1), Grads (2), Radians (3) """
    DegreesDecimal = None
    DegreesDms = None
    Grads = None
    Radians = None
    value__ = None


class LandXMLAttributeExportType(Enum):
    """ enum LandXMLAttributeExportType, values: Disabled (0), FullDescription (2), RawDescription (1) """
    Disabled = None
    FullDescription = None
    RawDescription = None
    value__ = None


class LandXMLConflictResolutionType(Enum):
    """ enum LandXMLConflictResolutionType, values: Append (2), Skip (0), Update (1) """
    Append = None
    Skip = None
    Update = None
    value__ = None


class LandXMLImperialUnitType(Enum):
    """ enum LandXMLImperialUnitType, values: Foot (30), Inch (31), Mile (44), Yard (33) """
    Foot = None
    Inch = None
    Mile = None
    value__ = None
    Yard = None


class LandXMLLinearUnits(Enum):
    """ enum LandXMLLinearUnits, values: InternationalFoot (30), SurveyFoot (54) """
    InternationalFoot = None
    SurveyFoot = None
    value__ = None


class LandXMLMetricUnitType(Enum):
    """ enum LandXMLMetricUnitType, values: CentiMeter (24), DeciMeter (23), KiloMeter (20), Meter (2), MilliMeter (25) """
    CentiMeter = None
    DeciMeter = None
    KiloMeter = None
    Meter = None
    MilliMeter = None
    value__ = None


class LandXMLPointDescriptionType(Enum):
    """ enum LandXMLPointDescriptionType, values: UseCodeThenDesc (2), UseCodeValue (0), UseDescThenCode (3), UseDescValue (1) """
    UseCodeThenDesc = None
    UseCodeValue = None
    UseDescThenCode = None
    UseDescValue = None
    value__ = None


class LandXMLSurfaceDataExportType(Enum):
    """ enum LandXMLSurfaceDataExportType, values: PointsAndFaces (1), PointsOnly (0) """
    PointsAndFaces = None
    PointsOnly = None
    value__ = None


class LandXMLSurfaceDataImportType(Enum):
    """ enum LandXMLSurfaceDataImportType, values: FullImport (1), QuickImport (0) """
    FullImport = None
    QuickImport = None
    value__ = None


class LocalCoordinateType(Enum):
    """ enum LocalCoordinateType, values: EastingNorthing (0), NorthingEasting (1), XY (2), YX (3) """
    EastingNorthing = None
    NorthingEasting = None
    value__ = None
    XY = None
    YX = None


class MapcheckAngleType(Enum):
    """ enum MapcheckAngleType, values: Angle (1), DeflectionAngle (2), Direction (0) """
    Angle = None
    DeflectionAngle = None
    Direction = None
    value__ = None


class MapcheckCurveDirectionType(Enum):
    """ enum MapcheckCurveDirectionType, values: Clockwise (0), CounterClockwise (1) """
    Clockwise = None
    CounterClockwise = None
    value__ = None


class MapcheckSideType(Enum):
    """ enum MapcheckSideType, values: Curve (1), Line (0) """
    Curve = None
    Line = None
    value__ = None


class MapcheckTraverseMethodType(Enum):
    """ enum MapcheckTraverseMethodType, values: AcrossChord (0), ThroughRadius (1) """
    AcrossChord = None
    ThroughRadius = None
    value__ = None


class ObjectLayerModifierType(Enum):
    """ enum ObjectLayerModifierType, values: None (0), Prefix (1), Suffix (2) """
    None = None
    Prefix = None
    Suffix = None
    value__ = None


class SectionViewAnchorType(Enum):
    """ enum SectionViewAnchorType, values: BottomCenter (7), BottomLeft (6), BottomRight (8), MiddleCenter (4), MiddleLeft (3), MiddleRight (5), TopCenter (1), TopLeft (0), TopRight (2) """
    BottomCenter = None
    BottomLeft = None
    BottomRight = None
    MiddleCenter = None
    MiddleLeft = None
    MiddleRight = None
    TopCenter = None
    TopLeft = None
    TopRight = None
    value__ = None


class SettingsAbbreviation(CivilWrapper<AcDbDatabase>):
    # no doc
    def Dispose(self):
        """ Dispose(self: CivilWrapper<AcDbDatabase>, A_0: bool) """
        pass

    AlignmentGeoPointEntityData = property(lambda self: object(), lambda self, v: None, lambda self: None)  # default
    """Get: AlignmentGeoPointEntityData(self: SettingsAbbreviation) -> SettingsAbbreviationAlignmentEnhanced

"""

    AlignmentGeoPointText = property(lambda self: object(), lambda self, v: None, lambda self: None)  # default
    """Get: AlignmentGeoPointText(self: SettingsAbbreviation) -> SettingsAbbreviationAlignment

"""

    Cant = property(lambda self: object(), lambda self, v: None, lambda self: None)  # default
    """Get: Cant(self: SettingsAbbreviation) -> SettingsAbbreviationCant

"""

    GeneralText = property(lambda self: object(), lambda self, v: None, lambda self: None)  # default
    """Get: GeneralText(self: SettingsAbbreviation) -> SettingsAbbreviationGeneral

"""

    Profile = property(lambda self: object(), lambda self, v: None, lambda self: None)  # default
    """Get: Profile(self: SettingsAbbreviation) -> SettingsAbbreviationProfile

"""

    Superelevation = property(lambda self: object(), lambda self, v: None, lambda self: None)  # default
    """Get: Superelevation(self: SettingsAbbreviation) -> SettingsAbbreviationSuperelevation

"""



class SettingsAbbreviationAlignment(TreeOidWrapper):
    # no doc
    def Dispose(self):
        """ Dispose(self: DisposableWrapper, A_0: bool) """
        pass

    def GetAlignmentAbbreviation(self, type):
        """ GetAlignmentAbbreviation(self: SettingsAbbreviationAlignment, type: AbbreviationAlignmentType) -> str """
        pass

    def SetAlignmentAbbreviation(self, type, value):
        """ SetAlignmentAbbreviation(self: SettingsAbbreviationAlignment, type: AbbreviationAlignmentType, value: str) """
        pass


class SettingsAbbreviationAlignmentEnhanced(TreeOidWrapper):
    # no doc
    def Dispose(self):
        """ Dispose(self: DisposableWrapper, A_0: bool) """
        pass

    def GetAlignmentEnhancedAbbreviation(self, type):
        """ GetAlignmentEnhancedAbbreviation(self: SettingsAbbreviationAlignmentEnhanced, type: AbbreviationAlignmentEnhancedType) -> str """
        pass

    def SetAlignmentEnhancedAbbreviation(self, type, newValue):
        """ SetAlignmentEnhancedAbbreviation(self: SettingsAbbreviationAlignmentEnhanced, type: AbbreviationAlignmentEnhancedType, newValue: str) """
        pass


class SettingsAbbreviationCant(TreeOidWrapper):
    # no doc
    def Dispose(self):
        """ Dispose(self: DisposableWrapper, A_0: bool) """
        pass

    def GetCantAbbreviation(self, type):
        """ GetCantAbbreviation(self: SettingsAbbreviationCant, type: AbbreviationCantType) -> str """
        pass

    def SetCantAbbreviation(self, type, newValue):
        """ SetCantAbbreviation(self: SettingsAbbreviationCant, type: AbbreviationCantType, newValue: str) """
        pass


class SettingsAbbreviationGeneral(TreeOidWrapper):
    # no doc
    def Dispose(self):
        """ Dispose(self: DisposableWrapper, A_0: bool) """
        pass

    Infinity = property(lambda self: object(), lambda self, v: None, lambda self: None)  # default
    """Get: Infinity(self: SettingsAbbreviationGeneral) -> str

Set: Infinity(self: SettingsAbbreviationGeneral) = value
"""

    Left = property(lambda self: object(), lambda self, v: None, lambda self: None)  # default
    """Get: Left(self: SettingsAbbreviationGeneral) -> str

Set: Left(self: SettingsAbbreviationGeneral) = value
"""

    Right = property(lambda self: object(), lambda self, v: None, lambda self: None)  # default
    """Get: Right(self: SettingsAbbreviationGeneral) -> str

Set: Right(self: SettingsAbbreviationGeneral) = value
"""



class SettingsAbbreviationProfile(TreeOidWrapper):
    # no doc
    def Dispose(self):
        """ Dispose(self: DisposableWrapper, A_0: bool) """
        pass

    def GetProfileAbbreviation(self, type):
        """ GetProfileAbbreviation(self: SettingsAbbreviationProfile, type: AbbreviationProfileType) -> str """
        pass

    def SetProfileAbbreviation(self, type, newValue):
        """ SetProfileAbbreviation(self: SettingsAbbreviationProfile, type: AbbreviationProfileType, newValue: str) """
        pass


class SettingsAbbreviationSuperelevation(TreeOidWrapper):
    # no doc
    def Dispose(self):
        """ Dispose(self: DisposableWrapper, A_0: bool) """
        pass

    def GetSuperelevationAbbreviation(self, type):
        """ GetSuperelevationAbbreviation(self: SettingsAbbreviationSuperelevation, type: AbbreviationSuperelevationType) -> str """
        pass

    def SetSuperelevationAbbreviation(self, type, newValue):
        """ SetSuperelevationAbbreviation(self: SettingsAbbreviationSuperelevation, type: AbbreviationSuperelevationType, newValue: str) """
        pass


class SettingsAmbient(TreeOidWrapper):
    # no doc
    def Dispose(self):
        """ Dispose(self: DisposableWrapper, A_0: bool) """
        pass

    @staticmethod # known case of __new__
    def __new__(self, *args): #cannot find CLR constructor
        """ __new__(cls: type, root: SettingsRoot, path: str) """
        pass

    Acceleration = property(lambda self: object(), lambda self, v: None, lambda self: None)  # default
    """Get: Acceleration(self: SettingsAmbient) -> SettingsAcceleration

"""

    Angle = property(lambda self: object(), lambda self, v: None, lambda self: None)  # default
    """Get: Angle(self: SettingsAmbient) -> SettingsAngle

"""

    Area = property(lambda self: object(), lambda self, v: None, lambda self: None)  # default
    """Get: Area(self: SettingsAmbient) -> SettingsArea

"""

    Coordinate = property(lambda self: object(), lambda self, v: None, lambda self: None)  # default
    """Get: Coordinate(self: SettingsAmbient) -> SettingsCoordinate

"""

    DegreeOfCurvature = property(lambda self: object(), lambda self, v: None, lambda self: None)  # default
    """Get: DegreeOfCurvature(self: SettingsAmbient) -> SettingsDegreeOfCurvature

"""

    Dimension = property(lambda self: object(), lambda self, v: None, lambda self: None)  # default
    """Get: Dimension(self: SettingsAmbient) -> SettingsDimension

"""

    Direction = property(lambda self: object(), lambda self, v: None, lambda self: None)  # default
    """Get: Direction(self: SettingsAmbient) -> SettingsDirection

"""

    Distance = property(lambda self: object(), lambda self, v: None, lambda self: None)  # default
    """Get: Distance(self: SettingsAmbient) -> SettingsDistance

"""

    Elevation = property(lambda self: object(), lambda self, v: None, lambda self: None)  # default
    """Get: Elevation(self: SettingsAmbient) -> SettingsElevation

"""

    General = property(lambda self: object(), lambda self, v: None, lambda self: None)  # default
    """Get: General(self: SettingsAmbient) -> SettingsGeneral

"""

    Grade = property(lambda self: object(), lambda self, v: None, lambda self: None)  # default
    """Get: Grade(self: SettingsAmbient) -> SettingsGrade

"""

    GradeSlope = property(lambda self: object(), lambda self, v: None, lambda self: None)  # default
    """Get: GradeSlope(self: SettingsAmbient) -> SettingsGradeSlope

"""

    GridCoordinate = property(lambda self: object(), lambda self, v: None, lambda self: None)  # default
    """Get: GridCoordinate(self: SettingsAmbient) -> SettingsGridCoordinate

"""

    Labeling = property(lambda self: object(), lambda self, v: None, lambda self: None)  # default
    """Get: Labeling(self: SettingsAmbient) -> SettingsLabeling

"""

    LatLong = property(lambda self: object(), lambda self, v: None, lambda self: None)  # default
    """Get: LatLong(self: SettingsAmbient) -> SettingsLatLong

"""

    Pressure = property(lambda self: object(), lambda self, v: None, lambda self: None)  # default
    """Get: Pressure(self: SettingsAmbient) -> SettingsPressure

"""

    Slope = property(lambda self: object(), lambda self, v: None, lambda self: None)  # default
    """Get: Slope(self: SettingsAmbient) -> SettingsSlope

"""

    Speed = property(lambda self: object(), lambda self, v: None, lambda self: None)  # default
    """Get: Speed(self: SettingsAmbient) -> SettingsSpeed

"""

    Station = property(lambda self: object(), lambda self, v: None, lambda self: None)  # default
    """Get: Station(self: SettingsAmbient) -> SettingsStation

"""

    Time = property(lambda self: object(), lambda self, v: None, lambda self: None)  # default
    """Get: Time(self: SettingsAmbient) -> SettingsTime

"""

    TransparentCommands = property(lambda self: object(), lambda self, v: None, lambda self: None)  # default
    """Get: TransparentCommands(self: SettingsAmbient) -> SettingsTransparentCommands

"""

    Unitless = property(lambda self: object(), lambda self, v: None, lambda self: None)  # default
    """Get: Unitless(self: SettingsAmbient) -> SettingsUnitless

"""

    Volume = property(lambda self: object(), lambda self, v: None, lambda self: None)  # default
    """Get: Volume(self: SettingsAmbient) -> SettingsVolume

"""


    SettingsAcceleration = None
    SettingsAngle = None
    SettingsArea = None
    SettingsCoordinate = None
    SettingsDegreeOfCurvature = None
    SettingsDimension = None
    SettingsDirection = None
    SettingsDistance = None
    SettingsElevation = None
    SettingsFormatNumber`1 = None
    SettingsGeneral = None
    SettingsGrade = None
    SettingsGradeSlope = None
    SettingsGridCoordinate = None
    SettingsLabeling = None
    SettingsLatLong = None
    SettingsPressure = None
    SettingsSlope = None
    SettingsSpeed = None
    SettingsStation = None
    SettingsTime = None
    SettingsTransparentCommands = None
    SettingsUnitFormatNumber`2 = None
    SettingsUnitless = None
    SettingsUnitlessNumber = None
    SettingsUnitNumber`1 = None
    SettingsVolume = None


class SettingsAlignment(SettingsAmbient):
    # no doc
    def Dispose(self):
        """ Dispose(self: DisposableWrapper, A_0: bool) """
        pass

    AutomaticWideningAroundCurves = property(lambda self: object(), lambda self, v: None, lambda self: None)  # default
    """Get: AutomaticWideningAroundCurves(self: SettingsAlignment) -> SettingsAutomaticWideningAroundCurves

"""

    CantOptions = property(lambda self: object(), lambda self, v: None, lambda self: None)  # default
    """Get: CantOptions(self: SettingsAlignment) -> SettingsCantOptions

"""

    ConstraintEditing = property(lambda self: object(), lambda self, v: None, lambda self: None)  # default
    """Get: ConstraintEditing(self: SettingsAlignment) -> SettingsConstraintEditing

"""

    CriteriaBasedDesignOptions = property(lambda self: object(), lambda self, v: None, lambda self: None)  # default
    """Get: CriteriaBasedDesignOptions(self: SettingsAlignment) -> SettingsCriteriaBasedDesignOptions

"""

    Data = property(lambda self: object(), lambda self, v: None, lambda self: None)  # default
    """Get: Data(self: SettingsAlignment) -> SettingsData

"""

    DefaultNameFormat = property(lambda self: object(), lambda self, v: None, lambda self: None)  # default
    """Get: DefaultNameFormat(self: SettingsAlignment) -> SettingsDefaultNameFormat

"""

    DynamicAlignmentHighlight = property(lambda self: object(), lambda self, v: None, lambda self: None)  # default
    """Get: DynamicAlignmentHighlight(self: SettingsAlignment) -> SettingsDynamicAlignmentHighlight

"""

    ImpliedPointOfIntersection = property(lambda self: object(), lambda self, v: None, lambda self: None)  # default
    """Get: ImpliedPointOfIntersection(self: SettingsAlignment) -> SettingsImpliedPointOfIntersection

"""

    RailOptions = property(lambda self: object(), lambda self, v: None, lambda self: None)  # default
    """Get: RailOptions(self: SettingsAlignment) -> SettingsRailAlignmentOptions

"""

    StationIndexing = property(lambda self: object(), lambda self, v: None, lambda self: None)  # default
    """Get: StationIndexing(self: SettingsAlignment) -> SettingsStationIndexing

"""

    StyleSettings = property(lambda self: object(), lambda self, v: None, lambda self: None)  # default
    """Get: StyleSettings(self: SettingsAlignment) -> SettingsStyles

"""

    SuperelevationOptions = property(lambda self: object(), lambda self, v: None, lambda self: None)  # default
    """Get: SuperelevationOptions(self: SettingsAlignment) -> SettingsSuperelevationOptions

"""


    SettingsAutomaticWideningAroundCurves = None
    SettingsCantOptions = None
    SettingsConstraintEditing = None
    SettingsCriteriaBasedDesignOptions = None
    SettingsData = None
    SettingsDefaultNameFormat = None
    SettingsDynamicAlignmentHighlight = None
    SettingsImpliedPointOfIntersection = None
    SettingsRailAlignmentOptions = None
    SettingsStationIndexing = None
    SettingsStyles = None
    SettingsSuperelevationOptions = None


class SettingsAssembly(SettingsAmbient):
    # no doc
    def Dispose(self):
        """ Dispose(self: DisposableWrapper, A_0: bool) """
        pass

    NameFormat = property(lambda self: object(), lambda self, v: None, lambda self: None)  # default
    """Get: NameFormat(self: SettingsAssembly) -> SettingsNameFormat

"""

    Styles = property(lambda self: object(), lambda self, v: None, lambda self: None)  # default
    """Get: Styles(self: SettingsAssembly) -> SettingsStyles

"""


    SettingsNameFormat = None
    SettingsStyles = None


class SettingsBuildingSite(SettingsAmbient):
    # no doc
    def Dispose(self):
        """ Dispose(self: DisposableWrapper, A_0: bool) """
        pass

    NameFormat = property(lambda self: object(), lambda self, v: None, lambda self: None)  # default
    """Get: NameFormat(self: SettingsBuildingSite) -> SettingsNameFormat

"""

    Styles = property(lambda self: object(), lambda self, v: None, lambda self: None)  # default
    """Get: Styles(self: SettingsBuildingSite) -> SettingsStyles

"""


    SettingsNameFormat = None
    SettingsStyles = None


class SettingsCantView(SettingsAmbient):
    # no doc
    def Dispose(self):
        """ Dispose(self: DisposableWrapper, A_0: bool) """
        pass

    NameFormat = property(lambda self: object(), lambda self, v: None, lambda self: None)  # default
    """Get: NameFormat(self: SettingsCantView) -> SettingsNameFormat

"""

    Styles = property(lambda self: object(), lambda self, v: None, lambda self: None)  # default
    """Get: Styles(self: SettingsCantView) -> SettingsStyles

"""


    SettingsNameFormat = None
    SettingsStyles = None


class SettingsCatchment(SettingsAmbient):
    # no doc
    def Dispose(self):
        """ Dispose(self: DisposableWrapper, A_0: bool) """
        pass

    NameTemplate = property(lambda self: object(), lambda self, v: None, lambda self: None)  # default
    """Get: NameTemplate(self: SettingsCatchment) -> PropertyString

"""

    Styles = property(lambda self: object(), lambda self, v: None, lambda self: None)  # default
    """Get: Styles(self: SettingsCatchment) -> SettingsStyles

"""


    SettingsStyles = None


class SettingsCmdAddAlignmentCurveTable(SettingsAlignment):
    # no doc
    def Dispose(self):
        """ Dispose(self: DisposableWrapper, A_0: bool) """
        pass

    TableCreation = property(lambda self: object(), lambda self, v: None, lambda self: None)  # default
    """Get: TableCreation(self: SettingsCmdAddAlignmentCurveTable) -> SettingsCmdTableCreation

"""


    SettingsCmdTableCreation = None


class SettingsCmdAddAlignmentLineTable(SettingsAlignment):
    # no doc
    def Dispose(self):
        """ Dispose(self: DisposableWrapper, A_0: bool) """
        pass

    TableCreation = property(lambda self: object(), lambda self, v: None, lambda self: None)  # default
    """Get: TableCreation(self: SettingsCmdAddAlignmentLineTable) -> SettingsCmdTableCreation

"""


    SettingsCmdTableCreation = None


class SettingsCmdAddAlignmentOffLbl(SettingsAlignment):
    # no doc
    def Dispose(self):
        """ Dispose(self: DisposableWrapper, A_0: bool) """
        pass


class SettingsCmdAddAlignmentOffXYLbl(SettingsAlignment):
    # no doc
    def Dispose(self):
        """ Dispose(self: DisposableWrapper, A_0: bool) """
        pass


class SettingsCmdAddAlignmentSegmentTable(SettingsAlignment):
    # no doc
    def Dispose(self):
        """ Dispose(self: DisposableWrapper, A_0: bool) """
        pass

    TableCreation = property(lambda self: object(), lambda self, v: None, lambda self: None)  # default
    """Get: TableCreation(self: SettingsCmdAddAlignmentSegmentTable) -> SettingsCmdTableCreation

"""


    SettingsCmdTableCreation = None


class SettingsCmdAddAlignmentSpiralTable(SettingsAlignment):
    # no doc
    def Dispose(self):
        """ Dispose(self: DisposableWrapper, A_0: bool) """
        pass

    TableCreation = property(lambda self: object(), lambda self, v: None, lambda self: None)  # default
    """Get: TableCreation(self: SettingsCmdAddAlignmentSpiralTable) -> SettingsCmdTableCreation

"""


    SettingsCmdTableCreation = None


class SettingsCmdAddAlignPointOfIntLbl(SettingsAlignment):
    # no doc
    def Dispose(self):
        """ Dispose(self: DisposableWrapper, A_0: bool) """
        pass


class SettingsCmdAddAlignPointOfIntLbls(SettingsAlignment):
    # no doc
    def Dispose(self):
        """ Dispose(self: DisposableWrapper, A_0: bool) """
        pass


class SettingsCmdAddAlignSegLbl(SettingsAlignment):
    # no doc
    def Dispose(self):
        """ Dispose(self: DisposableWrapper, A_0: bool) """
        pass


class SettingsCmdAddAlignSegLbls(SettingsAlignment):
    # no doc
    def Dispose(self):
        """ Dispose(self: DisposableWrapper, A_0: bool) """
        pass


class SettingsCmdAddAlignTagentLbl(SettingsAlignment):
    # no doc
    def Dispose(self):
        """ Dispose(self: DisposableWrapper, A_0: bool) """
        pass


class SettingsCmdAddAlignTagentLbls(SettingsAlignment):
    # no doc
    def Dispose(self):
        """ Dispose(self: DisposableWrapper, A_0: bool) """
        pass


class SettingsPressureNetwork(SettingsAmbient):
    # no doc
    def Dispose(self):
        """ Dispose(self: DisposableWrapper, A_0: bool) """
        pass

    Cover = property(lambda self: object(), lambda self, v: None, lambda self: None)  # default
    """Get: Cover(self: SettingsPressureNetwork) -> SettingsDepthOfCover

"""

    NameFormat = property(lambda self: object(), lambda self, v: None, lambda self: None)  # default
    """Get: NameFormat(self: SettingsPressureNetwork) -> SettingsNameFormat

"""

    ProfileLabelPlacement = property(lambda self: object(), lambda self, v: None, lambda self: None)  # default
    """Get: ProfileLabelPlacement(self: SettingsPressureNetwork) -> SettingsProfileLabelPlacement

"""

    SectionLabelPlacement = property(lambda self: object(), lambda self, v: None, lambda self: None)  # default
    """Get: SectionLabelPlacement(self: SettingsPressureNetwork) -> SettingsSectionLabelPlacement

"""

    Styles = property(lambda self: object(), lambda self, v: None, lambda self: None)  # default
    """Get: Styles(self: SettingsPressureNetwork) -> SettingsStyles

"""


    SettingsDepthOfCover = None
    SettingsNameFormat = None
    SettingsProfileLabelPlacement = None
    SettingsSectionLabelPlacement = None
    SettingsStyles = None


class SettingsCmdAddAppurtTable(SettingsPressureNetwork):
    # no doc
    def Dispose(self):
        """ Dispose(self: DisposableWrapper, A_0: bool) """
        pass

    TableCreation = property(lambda self: object(), lambda self, v: None, lambda self: None)  # default
    """Get: TableCreation(self: SettingsCmdAddAppurtTable) -> SettingsCmdTableCreation

"""


    SettingsCmdTableCreation = None


class SettingsSurface(SettingsAmbient):
    # no doc
    def Dispose(self):
        """ Dispose(self: DisposableWrapper, A_0: bool) """
        pass

    ContourLabeling = property(lambda self: object(), lambda self, v: None, lambda self: None)  # default
    """Get: ContourLabeling(self: SettingsSurface) -> SettingsContourLabeling

"""

    Defaults = property(lambda self: object(), lambda self, v: None, lambda self: None)  # default
    """Get: Defaults(self: SettingsSurface) -> SettingsDefaults

"""

    Styles = property(lambda self: object(), lambda self, v: None, lambda self: None)  # default
    """Get: Styles(self: SettingsSurface) -> SettingsStyles

"""


    SettingsContourLabeling = None
    SettingsDefaults = None
    SettingsStyles = None


class SettingsCmdAddContourLabeling(SettingsSurface):
    # no doc
    def Dispose(self):
        """ Dispose(self: DisposableWrapper, A_0: bool) """
        pass


class SettingsCmdAddContourLabelingGroup(SettingsSurface):
    # no doc
    def Dispose(self):
        """ Dispose(self: DisposableWrapper, A_0: bool) """
        pass

    AddContourLabeling = property(lambda self: object(), lambda self, v: None, lambda self: None)  # default
    """Get: AddContourLabeling(self: SettingsCmdAddContourLabelingGroup) -> SettingsCmdAddContourLabeling

"""


    SettingsCmdAddContourLabeling = None


class SettingsCmdAddContourLabelingSingle(SettingsSurface):
    # no doc
    def Dispose(self):
        """ Dispose(self: DisposableWrapper, A_0: bool) """
        pass


class SettingsCmdAddFittingTable(SettingsPressureNetwork):
    # no doc
    def Dispose(self):
        """ Dispose(self: DisposableWrapper, A_0: bool) """
        pass

    TableCreation = property(lambda self: object(), lambda self, v: None, lambda self: None)  # default
    """Get: TableCreation(self: SettingsCmdAddFittingTable) -> SettingsCmdTableCreation

"""


    SettingsCmdTableCreation = None


class SettingsIntersection(SettingsAmbient):
    # no doc
    def Dispose(self):
        """ Dispose(self: DisposableWrapper, A_0: bool) """
        pass

    NameFormat = property(lambda self: object(), lambda self, v: None, lambda self: None)  # default
    """Get: NameFormat(self: SettingsIntersection) -> SettingsNameFormat

"""

    Styles = property(lambda self: object(), lambda self, v: None, lambda self: None)  # default
    """Get: Styles(self: SettingsIntersection) -> SettingsStyles

"""


    SettingsNameFormat = None
    SettingsStyles = None


class SettingsCmdAddIntersectionLabel(SettingsIntersection):
    # no doc
    def Dispose(self):
        """ Dispose(self: DisposableWrapper, A_0: bool) """
        pass


class SettingsGeneral(SettingsAmbient):
    # no doc
    def Dispose(self):
        """ Dispose(self: DisposableWrapper, A_0: bool) """
        pass

    Styles = property(lambda self: object(), lambda self, v: None, lambda self: None)  # default
    """Get: Styles(self: SettingsGeneral) -> SettingsStyles

"""


    SettingsStyles = None


class SettingsCmdAddLineBetweenPoints(SettingsGeneral):
    # no doc
    def Dispose(self):
        """ Dispose(self: DisposableWrapper, A_0: bool) """
        pass


class SettingsQuantityTakeoff(SettingsAmbient):
    # no doc
    def Dispose(self):
        """ Dispose(self: DisposableWrapper, A_0: bool) """
        pass

    NameFormat = property(lambda self: object(), lambda self, v: None, lambda self: None)  # default
    """Get: NameFormat(self: SettingsQuantityTakeoff) -> SettingsNameFormat

"""

    Styles = property(lambda self: object(), lambda self, v: None, lambda self: None)  # default
    """Get: Styles(self: SettingsQuantityTakeoff) -> SettingsStyles

"""


    SettingsNameFormat = None
    SettingsStyles = None


class SettingsCmdAddMaterialVolumeTable(SettingsQuantityTakeoff):
    # no doc
    def Dispose(self):
        """ Dispose(self: DisposableWrapper, A_0: bool) """
        pass

    TableCreation = property(lambda self: object(), lambda self, v: None, lambda self: None)  # default
    """Get: TableCreation(self: SettingsCmdAddMaterialVolumeTable) -> SettingsCmdTableCreation

"""


    SettingsCmdTableCreation = None


class SettingsPipeNetwork(SettingsAmbient):
    # no doc
    def Dispose(self):
        """ Dispose(self: DisposableWrapper, A_0: bool) """
        pass

    Default = property(lambda self: object(), lambda self, v: None, lambda self: None)  # default
    """Get: Default(self: SettingsPipeNetwork) -> SettingsDefault

"""

    NameFormat = property(lambda self: object(), lambda self, v: None, lambda self: None)  # default
    """Get: NameFormat(self: SettingsPipeNetwork) -> SettingsNameFormat

"""

    ProfileLabelPlacement = property(lambda self: object(), lambda self, v: None, lambda self: None)  # default
    """Get: ProfileLabelPlacement(self: SettingsPipeNetwork) -> SettingsProfileLabelPlacement

"""

    Rules = property(lambda self: object(), lambda self, v: None, lambda self: None)  # default
    """Get: Rules(self: SettingsPipeNetwork) -> SettingsRules

"""

    SectionLabelPlacement = property(lambda self: object(), lambda self, v: None, lambda self: None)  # default
    """Get: SectionLabelPlacement(self: SettingsPipeNetwork) -> SettingsSectionLabelPlacement

"""

    StormSewersMigration = property(lambda self: object(), lambda self, v: None, lambda self: None)  # default
    """Get: StormSewersMigration(self: SettingsPipeNetwork) -> SettingsStormSewersMigration

"""

    Styles = property(lambda self: object(), lambda self, v: None, lambda self: None)  # default
    """Get: Styles(self: SettingsPipeNetwork) -> SettingsStyles

"""


    SettingsDefault = None
    SettingsNameFormat = None
    SettingsProfileLabelPlacement = None
    SettingsRules = None
    SettingsSectionLabelPlacement = None
    SettingsStormSewersMigration = None
    SettingsStyles = None


class SettingsCmdAddNetworkPartPlanLabel(SettingsPipeNetwork):
    # no doc
    def Dispose(self):
        """ Dispose(self: DisposableWrapper, A_0: bool) """
        pass


class SettingsCmdAddNetworkPartProfLabel(SettingsPipeNetwork):
    # no doc
    def Dispose(self):
        """ Dispose(self: DisposableWrapper, A_0: bool) """
        pass


class SettingsCmdAddNetworkPartSectLabel(SettingsPipeNetwork):
    # no doc
    def Dispose(self):
        """ Dispose(self: DisposableWrapper, A_0: bool) """
        pass


class SettingsCmdAddNetworkPartsToProf(SettingsPipeNetwork):
    # no doc
    def Dispose(self):
        """ Dispose(self: DisposableWrapper, A_0: bool) """
        pass


class SettingsCmdAddNetworkPipeTable(SettingsPipeNetwork):
    # no doc
    def Dispose(self):
        """ Dispose(self: DisposableWrapper, A_0: bool) """
        pass

    TableCreation = property(lambda self: object(), lambda self, v: None, lambda self: None)  # default
    """Get: TableCreation(self: SettingsCmdAddNetworkPipeTable) -> SettingsCmdTableCreation

"""


    SettingsCmdTableCreation = None


class SettingsCmdAddNetworkPlanLabels(SettingsPipeNetwork):
    # no doc
    def Dispose(self):
        """ Dispose(self: DisposableWrapper, A_0: bool) """
        pass


class SettingsCmdAddNetworkProfLabels(SettingsPipeNetwork):
    # no doc
    def Dispose(self):
        """ Dispose(self: DisposableWrapper, A_0: bool) """
        pass


class SettingsCmdAddNetworkSectLabels(SettingsPipeNetwork):
    # no doc
    def Dispose(self):
        """ Dispose(self: DisposableWrapper, A_0: bool) """
        pass


class SettingsCmdAddNetworkStructTable(SettingsPipeNetwork):
    # no doc
    def Dispose(self):
        """ Dispose(self: DisposableWrapper, A_0: bool) """
        pass

    TableCreation = property(lambda self: object(), lambda self, v: None, lambda self: None)  # default
    """Get: TableCreation(self: SettingsCmdAddNetworkStructTable) -> SettingsCmdTableCreation

"""


    SettingsCmdTableCreation = None


class SettingsCmdAddNoteLabel(SettingsGeneral):
    # no doc
    def Dispose(self):
        """ Dispose(self: DisposableWrapper, A_0: bool) """
        pass


class SettingsParcel(SettingsAmbient):
    # no doc
    def Dispose(self):
        """ Dispose(self: DisposableWrapper, A_0: bool) """
        pass

    Styles = property(lambda self: object(), lambda self, v: None, lambda self: None)  # default
    """Get: Styles(self: SettingsParcel) -> SettingsStyles

"""


    SettingsStyles = None


class SettingsCmdAddParcelAreaLabel(SettingsParcel):
    # no doc
    def Dispose(self):
        """ Dispose(self: DisposableWrapper, A_0: bool) """
        pass


class SettingsCmdAddParcelCurveTable(SettingsParcel):
    # no doc
    def Dispose(self):
        """ Dispose(self: DisposableWrapper, A_0: bool) """
        pass

    TableCreation = property(lambda self: object(), lambda self, v: None, lambda self: None)  # default
    """Get: TableCreation(self: SettingsCmdAddParcelCurveTable) -> SettingsCmdTableCreation

"""


    SettingsCmdTableCreation = None


class SettingsCmdAddParcelLineLabel(SettingsParcel):
    # no doc
    def Dispose(self):
        """ Dispose(self: DisposableWrapper, A_0: bool) """
        pass


class SettingsCmdAddParcelLineTable(SettingsParcel):
    # no doc
    def Dispose(self):
        """ Dispose(self: DisposableWrapper, A_0: bool) """
        pass

    TableCreation = property(lambda self: object(), lambda self, v: None, lambda self: None)  # default
    """Get: TableCreation(self: SettingsCmdAddParcelLineTable) -> SettingsCmdTableCreation

"""


    SettingsCmdTableCreation = None


class SettingsCmdAddParcelSegmentLabels(SettingsParcel):
    # no doc
    def Dispose(self):
        """ Dispose(self: DisposableWrapper, A_0: bool) """
        pass

    Options = property(lambda self: object(), lambda self, v: None, lambda self: None)  # default
    """Get: Options(self: SettingsCmdAddParcelSegmentLabels) -> SettingsCmdOptions

"""


    SettingsCmdOptions = None


class SettingsCmdAddParcelSegmentTable(SettingsParcel):
    # no doc
    def Dispose(self):
        """ Dispose(self: DisposableWrapper, A_0: bool) """
        pass

    TableCreation = property(lambda self: object(), lambda self, v: None, lambda self: None)  # default
    """Get: TableCreation(self: SettingsCmdAddParcelSegmentTable) -> SettingsCmdTableCreation

"""


    SettingsCmdTableCreation = None


class SettingsCmdAddParcelTable(SettingsParcel):
    # no doc
    def Dispose(self):
        """ Dispose(self: DisposableWrapper, A_0: bool) """
        pass

    TableCreation = property(lambda self: object(), lambda self, v: None, lambda self: None)  # default
    """Get: TableCreation(self: SettingsCmdAddParcelTable) -> SettingsCmdTableCreation

"""


    SettingsCmdTableCreation = None


class SettingsPointCloud(SettingsAmbient):
    # no doc
    def Dispose(self):
        """ Dispose(self: DisposableWrapper, A_0: bool) """
        pass

    DefaultNameFormat = property(lambda self: object(), lambda self, v: None, lambda self: None)  # default
    """Get: DefaultNameFormat(self: SettingsPointCloud) -> SettingsDefaultNameFormat

"""

    StyleSettings = property(lambda self: object(), lambda self, v: None, lambda self: None)  # default
    """Get: StyleSettings(self: SettingsPointCloud) -> SettingsStyles

"""


    SettingsDefaultNameFormat = None
    SettingsStyles = None


class SettingsCmdAddPointCloudPoints(SettingsPointCloud):
    # no doc
    def Dispose(self):
        """ Dispose(self: DisposableWrapper, A_0: bool) """
        pass

    DefaultFileFormat = property(lambda self: object(), lambda self, v: None, lambda self: None)  # default
    """Get: DefaultFileFormat(self: SettingsCmdAddPointCloudPoints) -> PropertyEnum[PointCloudDefaultFileExtensionType]

"""



class SettingsCmdAddPointsToSurface(SettingsPointCloud):
    # no doc
    def Dispose(self):
        """ Dispose(self: DisposableWrapper, A_0: bool) """
        pass

    MidOrdinateDistance = property(lambda self: object(), lambda self, v: None, lambda self: None)  # default
    """Get: MidOrdinateDistance(self: SettingsCmdAddPointsToSurface) -> PropertyDouble

"""

    RegionOption = property(lambda self: object(), lambda self, v: None, lambda self: None)  # default
    """Get: RegionOption(self: SettingsCmdAddPointsToSurface) -> PropertyEnum[PointCloudRegionType]

"""

    SurfaceOption = property(lambda self: object(), lambda self, v: None, lambda self: None)  # default
    """Get: SurfaceOption(self: SettingsCmdAddPointsToSurface) -> PropertyEnum[PointCloudSurfaceType]

"""



class SettingsPoint(SettingsAmbient):
    # no doc
    def Dispose(self):
        """ Dispose(self: DisposableWrapper, A_0: bool) """
        pass

    NameFormat = property(lambda self: object(), lambda self, v: None, lambda self: None)  # default
    """Get: NameFormat(self: SettingsPoint) -> SettingsNameFormat

"""

    Styles = property(lambda self: object(), lambda self, v: None, lambda self: None)  # default
    """Get: Styles(self: SettingsPoint) -> SettingsStyles

"""

    UpdatePoints = property(lambda self: object(), lambda self, v: None, lambda self: None)  # default
    """Get: UpdatePoints(self: SettingsPoint) -> SettingsUpdatePoints

"""


    SettingsNameFormat = None
    SettingsStyles = None
    SettingsUpdatePoints = None


class SettingsCmdAddPointTable(SettingsPoint):
    # no doc
    def Dispose(self):
        """ Dispose(self: DisposableWrapper, A_0: bool) """
        pass

    TableCreation = property(lambda self: object(), lambda self, v: None, lambda self: None)  # default
    """Get: TableCreation(self: SettingsCmdAddPointTable) -> SettingsCmdTableCreation

"""


    SettingsCmdTableCreation = None


class SettingsCmdAddPressurePartPlanLabel(SettingsPressureNetwork):
    # no doc
    def Dispose(self):
        """ Dispose(self: DisposableWrapper, A_0: bool) """
        pass


class SettingsCmdAddPressurePartProfLabel(SettingsPressureNetwork):
    # no doc
    def Dispose(self):
        """ Dispose(self: DisposableWrapper, A_0: bool) """
        pass


class SettingsCmdAddPressurePartsToProf(SettingsPressureNetwork):
    # no doc
    def Dispose(self):
        """ Dispose(self: DisposableWrapper, A_0: bool) """
        pass


class SettingsCmdAddPressurePipeTable(SettingsPressureNetwork):
    # no doc
    def Dispose(self):
        """ Dispose(self: DisposableWrapper, A_0: bool) """
        pass

    TableCreation = property(lambda self: object(), lambda self, v: None, lambda self: None)  # default
    """Get: TableCreation(self: SettingsCmdAddPressurePipeTable) -> SettingsCmdTableCreation

"""


    SettingsCmdTableCreation = None


class SettingsCmdAddPressurePlanLabels(SettingsPressureNetwork):
    # no doc
    def Dispose(self):
        """ Dispose(self: DisposableWrapper, A_0: bool) """
        pass


class SettingsCmdAddPressureProfLabels(SettingsPressureNetwork):
    # no doc
    def Dispose(self):
        """ Dispose(self: DisposableWrapper, A_0: bool) """
        pass


class SettingsProfileView(SettingsAmbient):
    # no doc
    def Dispose(self):
        """ Dispose(self: DisposableWrapper, A_0: bool) """
        pass

    Creation = property(lambda self: object(), lambda self, v: None, lambda self: None)  # default
    """Get: Creation(self: SettingsProfileView) -> SettingsCreation

"""

    NameFormat = property(lambda self: object(), lambda self, v: None, lambda self: None)  # default
    """Get: NameFormat(self: SettingsProfileView) -> SettingsNameFormat

"""

    ProjectionLabelPlacement = property(lambda self: object(), lambda self, v: None, lambda self: None)  # default
    """Get: ProjectionLabelPlacement(self: SettingsProfileView) -> SettingsProjectionLabelPlacement

"""

    SplitOptions = property(lambda self: object(), lambda self, v: None, lambda self: None)  # default
    """Get: SplitOptions(self: SettingsProfileView) -> SettingsSplitOptions

"""

    StackedOptions = property(lambda self: object(), lambda self, v: None, lambda self: None)  # default
    """Get: StackedOptions(self: SettingsProfileView) -> SettingsStackedOptions

"""

    Styles = property(lambda self: object(), lambda self, v: None, lambda self: None)  # default
    """Get: Styles(self: SettingsProfileView) -> SettingsStyles

"""


    SettingsCreation = None
    SettingsNameFormat = None
    SettingsProjectionLabelPlacement = None
    SettingsSplitOptions = None
    SettingsStackedOptions = None
    SettingsStyles = None


class SettingsCmdAddProfileViewDepthLbl(SettingsProfileView):
    # no doc
    def Dispose(self):
        """ Dispose(self: DisposableWrapper, A_0: bool) """
        pass


class SettingsCmdAddProfileViewStaElevLbl(SettingsProfileView):
    # no doc
    def Dispose(self):
        """ Dispose(self: DisposableWrapper, A_0: bool) """
        pass


class SettingsSectionView(SettingsAmbient):
    # no doc
    def Dispose(self):
        """ Dispose(self: DisposableWrapper, A_0: bool) """
        pass

    NameFormat = property(lambda self: object(), lambda self, v: None, lambda self: None)  # default
    """Get: NameFormat(self: SettingsSectionView) -> SettingsNameFormat

"""

    ProjectionLabelPlacement = property(lambda self: object(), lambda self, v: None, lambda self: None)  # default
    """Get: ProjectionLabelPlacement(self: SettingsSectionView) -> SettingsProjectionLabelPlacement

"""

    SectionViewCreation = property(lambda self: object(), lambda self, v: None, lambda self: None)  # default
    """Get: SectionViewCreation(self: SettingsSectionView) -> SettingsSectionViewCreation

"""

    Styles = property(lambda self: object(), lambda self, v: None, lambda self: None)  # default
    """Get: Styles(self: SettingsSectionView) -> SettingsStyles

"""


    SettingsNameFormat = None
    SettingsProjectionLabelPlacement = None
    SettingsSectionViewCreation = None
    SettingsStyles = None


class SettingsCmdAddSectionViewGradeLbl(SettingsSectionView):
    # no doc
    def Dispose(self):
        """ Dispose(self: DisposableWrapper, A_0: bool) """
        pass


class SettingsCmdAddSectionViewOffElevLbl(SettingsSectionView):
    # no doc
    def Dispose(self):
        """ Dispose(self: DisposableWrapper, A_0: bool) """
        pass


class SettingsCmdAddSegmentLabel(SettingsGeneral):
    # no doc
    def Dispose(self):
        """ Dispose(self: DisposableWrapper, A_0: bool) """
        pass


class SettingsCmdAddSegmentLabels(SettingsGeneral):
    # no doc
    def Dispose(self):
        """ Dispose(self: DisposableWrapper, A_0: bool) """
        pass


class SettingsCmdAddSpanningPipePlanLabel(SettingsPipeNetwork):
    # no doc
    def Dispose(self):
        """ Dispose(self: DisposableWrapper, A_0: bool) """
        pass


class SettingsCmdAddSpanningPipeProfLabel(SettingsPipeNetwork):
    # no doc
    def Dispose(self):
        """ Dispose(self: DisposableWrapper, A_0: bool) """
        pass


class SettingsCmdAddSpotElevLabelsOnGrid(SettingsSurface):
    # no doc
    def Dispose(self):
        """ Dispose(self: DisposableWrapper, A_0: bool) """
        pass


class SettingsCmdAddSurfaceBoundaries(SettingsSurface):
    # no doc
    def Dispose(self):
        """ Dispose(self: DisposableWrapper, A_0: bool) """
        pass

    DataOptions = property(lambda self: object(), lambda self, v: None, lambda self: None)  # default
    """Get: DataOptions(self: SettingsCmdAddSurfaceBoundaries) -> SettingsCmdAddDataOptions

"""


    SettingsCmdAddDataOptions = None


class SettingsCmdAddSurfaceBreaklines(SettingsSurface):
    # no doc
    def Dispose(self):
        """ Dispose(self: DisposableWrapper, A_0: bool) """
        pass

    DataOptions = property(lambda self: object(), lambda self, v: None, lambda self: None)  # default
    """Get: DataOptions(self: SettingsCmdAddSurfaceBreaklines) -> SettingsCmdAddDataOptions

"""


    SettingsCmdAddDataOptions = None


class SettingsCmdAddSurfaceContours(SettingsSurface):
    # no doc
    def Dispose(self):
        """ Dispose(self: DisposableWrapper, A_0: bool) """
        pass

    AddDataOptions = property(lambda self: object(), lambda self, v: None, lambda self: None)  # default
    """Get: AddDataOptions(self: SettingsCmdAddSurfaceContours) -> SettingsCmdAddDataOptions

"""


    SettingsCmdAddDataOptions = None


class SettingsCmdAddSurfaceDemFile(SettingsSurface):
    # no doc
    def Dispose(self):
        """ Dispose(self: DisposableWrapper, A_0: bool) """
        pass

    ImportOptions = property(lambda self: object(), lambda self, v: None, lambda self: None)  # default
    """Get: ImportOptions(self: SettingsCmdAddSurfaceDemFile) -> SettingsCmdImportOptions

"""


    SettingsCmdImportOptions = None


class SettingsCmdAddSurfaceDrawingObjects(SettingsSurface):
    # no doc
    def Dispose(self):
        """ Dispose(self: DisposableWrapper, A_0: bool) """
        pass

    DataOptions = property(lambda self: object(), lambda self, v: None, lambda self: None)  # default
    """Get: DataOptions(self: SettingsCmdAddSurfaceDrawingObjects) -> SettingsCmdAddDataOptions

"""


    SettingsCmdAddDataOptions = None


class SettingsCmdAddSurfaceFigSurveyQuery(SettingsSurface):
    # no doc
    def Dispose(self):
        """ Dispose(self: DisposableWrapper, A_0: bool) """
        pass

    DataOptions = property(lambda self: object(), lambda self, v: None, lambda self: None)  # default
    """Get: DataOptions(self: SettingsCmdAddSurfaceFigSurveyQuery) -> SettingsCmdAddDataOptions

"""


    SettingsCmdAddDataOptions = None


class SettingsCmdAddSurfacePointSurveyQuery(SettingsSurface):
    # no doc
    def Dispose(self):
        """ Dispose(self: DisposableWrapper, A_0: bool) """
        pass


class SettingsCmdAddSurfaceSlopeLabel(SettingsSurface):
    # no doc
    def Dispose(self):
        """ Dispose(self: DisposableWrapper, A_0: bool) """
        pass


class SettingsCmdAddSurfaceSpotElevLabel(SettingsSurface):
    # no doc
    def Dispose(self):
        """ Dispose(self: DisposableWrapper, A_0: bool) """
        pass


class SettingsSurvey(SettingsAmbient):
    # no doc
    def Dispose(self):
        """ Dispose(self: DisposableWrapper, A_0: bool) """
        pass

    Styles = property(lambda self: object(), lambda self, v: None, lambda self: None)  # default
    """Get: Styles(self: SettingsSurvey) -> SettingsStyles

"""


    SettingsStyles = None


class SettingsCmdAddSvFigureLabel(SettingsSurvey):
    # no doc
    def Dispose(self):
        """ Dispose(self: DisposableWrapper, A_0: bool) """
        pass


class SettingsCmdAddSvFigureSegmentLabel(SettingsSurvey):
    # no doc
    def Dispose(self):
        """ Dispose(self: DisposableWrapper, A_0: bool) """
        pass


class SettingsCmdAddSvFigureSegmentLabels(SettingsSurvey):
    # no doc
    def Dispose(self):
        """ Dispose(self: DisposableWrapper, A_0: bool) """
        pass


class SettingsCmdAddTotalVolumeTable(SettingsQuantityTakeoff):
    # no doc
    def Dispose(self):
        """ Dispose(self: DisposableWrapper, A_0: bool) """
        pass

    TableCreation = property(lambda self: object(), lambda self, v: None, lambda self: None)  # default
    """Get: TableCreation(self: SettingsCmdAddTotalVolumeTable) -> SettingsCmdTableCreation

"""


    SettingsCmdTableCreation = None


class SettingsCmdAddWidening(SettingsAlignment):
    # no doc
    def Dispose(self):
        """ Dispose(self: DisposableWrapper, A_0: bool) """
        pass

    LinearTransitionAroundCurves = property(lambda self: object(), lambda self, v: None, lambda self: None)  # default
    """Get: LinearTransitionAroundCurves(self: SettingsCmdAddWidening) -> SettingsCmdLinearTransitionAroundCurves

"""

    WideningOptions = property(lambda self: object(), lambda self, v: None, lambda self: None)  # default
    """Get: WideningOptions(self: SettingsCmdAddWidening) -> SettingsCmdWideningOptions

"""


    SettingsCmdLinearTransitionAroundCurves = None
    SettingsCmdWideningOptions = None


class SettingsCmdAssignPayItemToArea(SettingsQuantityTakeoff):
    # no doc
    def Dispose(self):
        """ Dispose(self: DisposableWrapper, A_0: bool) """
        pass

    AssignPayItemToAreaOption = property(lambda self: object(), lambda self, v: None, lambda self: None)  # default
    """Get: AssignPayItemToAreaOption(self: SettingsCmdAssignPayItemToArea) -> SettingsCmdAssignPayItemToAreaOptions

"""


    SettingsCmdAssignPayItemToAreaOptions = None


class SettingsCmdCatchmentArea(SettingsSurface):
    # no doc
    def Dispose(self):
        """ Dispose(self: DisposableWrapper, A_0: bool) """
        pass

    DischargePointStyle = property(lambda self: object(), lambda self, v: None, lambda self: None)  # default
    """Get: DischargePointStyle(self: SettingsCmdCatchmentArea) -> PropertyString

"""

    DischargePointStyleId = property(lambda self: object(), lambda self, v: None, lambda self: None)  # default
    """Get: DischargePointStyleId(self: SettingsCmdCatchmentArea) -> PropertyObjectId

"""

    DisplayDisChargePoint = property(lambda self: object(), lambda self, v: None, lambda self: None)  # default
    """Get: DisplayDisChargePoint(self: SettingsCmdCatchmentArea) -> PropertyBoolean

"""

    Layer = property(lambda self: object(), lambda self, v: None, lambda self: None)  # default
    """Get: Layer(self: SettingsCmdCatchmentArea) -> PropertyLayer

"""

    ObjectType = property(lambda self: object(), lambda self, v: None, lambda self: None)  # default
    """Get: ObjectType(self: SettingsCmdCatchmentArea) -> PropertyEnum[CatchmentObjectType]

"""



class SettingsCmdComputeMaterials(SettingsQuantityTakeoff):
    # no doc
    def Dispose(self):
        """ Dispose(self: DisposableWrapper, A_0: bool) """
        pass

    DefineMaterialOption = property(lambda self: object(), lambda self, v: None, lambda self: None)  # default
    """Get: DefineMaterialOption(self: SettingsCmdComputeMaterials) -> SettingsCmdDefineMaterial

"""


    SettingsCmdDefineMaterial = None


class SettingsCmdConvertPointstoSdskPoints(SettingsPoint):
    # no doc
    def Dispose(self):
        """ Dispose(self: DisposableWrapper, A_0: bool) """
        pass

    Layer = property(lambda self: object(), lambda self, v: None, lambda self: None)  # default
    """Get: Layer(self: SettingsCmdConvertPointstoSdskPoints) -> SettingsCmdLayer

"""


    SettingsCmdLayer = None


class SettingsCorridor(SettingsAmbient):
    # no doc
    def Dispose(self):
        """ Dispose(self: DisposableWrapper, A_0: bool) """
        pass

    NameFormat = property(lambda self: object(), lambda self, v: None, lambda self: None)  # default
    """Get: NameFormat(self: SettingsCorridor) -> SettingsNameFormat

"""

    RegionHighlightGraphics = property(lambda self: object(), lambda self, v: None, lambda self: None)  # default
    """Get: RegionHighlightGraphics(self: SettingsCorridor) -> SettingsRegionHighlightGraphics

"""

    Styles = property(lambda self: object(), lambda self, v: None, lambda self: None)  # default
    """Get: Styles(self: SettingsCorridor) -> SettingsStyles

"""


    SettingsNameFormat = None
    SettingsRegionHighlightGraphics = None
    SettingsStyles = None


class SettingsCmdCorridorExtractSurfaces(SettingsCorridor):
    # no doc
    def Dispose(self):
        """ Dispose(self: DisposableWrapper, A_0: bool) """
        pass


class SettingsCmdCreateAlignFromCorridor(SettingsCorridor):
    # no doc
    def Dispose(self):
        """ Dispose(self: DisposableWrapper, A_0: bool) """
        pass

    AlignmentTypeOption = property(lambda self: object(), lambda self, v: None, lambda self: None)  # default
    """Get: AlignmentTypeOption(self: SettingsCmdCreateAlignFromCorridor) -> SettingsCmdAlignmentTypeOption

"""

    CriteriaBasedDesignOptions = property(lambda self: object(), lambda self, v: None, lambda self: None)  # default
    """Get: CriteriaBasedDesignOptions(self: SettingsCmdCreateAlignFromCorridor) -> SettingsCmdCriteriaBasedDesignOptions

"""

    ProfileCreationOption = property(lambda self: object(), lambda self, v: None, lambda self: None)  # default
    """Get: ProfileCreationOption(self: SettingsCmdCreateAlignFromCorridor) -> SettingsCmdProfileCreationOption

"""


    SettingsCmdAlignmentTypeOption = None
    SettingsCmdCriteriaBasedDesignOptions = None
    SettingsCmdProfileCreationOption = None


class SettingsCmdCreateAlignFromNetwork(SettingsPipeNetwork):
    # no doc
    def Dispose(self):
        """ Dispose(self: DisposableWrapper, A_0: bool) """
        pass

    AlignmentTypeOption = property(lambda self: object(), lambda self, v: None, lambda self: None)  # default
    """Get: AlignmentTypeOption(self: SettingsCmdCreateAlignFromNetwork) -> SettingsCmdAlignmentTypeOption

"""


    SettingsCmdAlignmentTypeOption = None


class SettingsCmdCreateAlignFromPressureNW(SettingsPressureNetwork):
    # no doc
    def Dispose(self):
        """ Dispose(self: DisposableWrapper, A_0: bool) """
        pass

    AlignmentTypeOption = property(lambda self: object(), lambda self, v: None, lambda self: None)  # default
    """Get: AlignmentTypeOption(self: SettingsCmdCreateAlignFromPressureNW) -> SettingsCmdAlignmentTypeOption

"""


    SettingsCmdAlignmentTypeOption = None


class SettingsCmdCreateAlignmentEntities(SettingsAlignment):
    # no doc
    def Dispose(self):
        """ Dispose(self: DisposableWrapper, A_0: bool) """
        pass

    AlignmentTypeOption = property(lambda self: object(), lambda self, v: None, lambda self: None)  # default
    """Get: AlignmentTypeOption(self: SettingsCmdCreateAlignmentEntities) -> SettingsCmdAlignmentTypeOption

"""

    CreateFromEntities = property(lambda self: object(), lambda self, v: None, lambda self: None)  # default
    """Get: CreateFromEntities(self: SettingsCmdCreateAlignmentEntities) -> SettingsCmdCreateFromEntities

"""


    SettingsCmdAlignmentTypeOption = None
    SettingsCmdCreateFromEntities = None


class SettingsCmdCreateAlignmentLayout(SettingsAlignment):
    # no doc
    def Dispose(self):
        """ Dispose(self: DisposableWrapper, A_0: bool) """
        pass

    AlignmentTypeOption = property(lambda self: object(), lambda self, v: None, lambda self: None)  # default
    """Get: AlignmentTypeOption(self: SettingsCmdCreateAlignmentLayout) -> SettingsCmdAlignmentTypeOption

"""

    CurveAndSpiralSettings = property(lambda self: object(), lambda self, v: None, lambda self: None)  # default
    """Get: CurveAndSpiralSettings(self: SettingsCmdCreateAlignmentLayout) -> SettingsCmdCurveAndSpiralSettings

"""

    CurveTessellationOption = property(lambda self: object(), lambda self, v: None, lambda self: None)  # default
    """Get: CurveTessellationOption(self: SettingsCmdCreateAlignmentLayout) -> SettingsCmdCurveTessellationOption

"""

    RegressionGraphOption = property(lambda self: object(), lambda self, v: None, lambda self: None)  # default
    """Get: RegressionGraphOption(self: SettingsCmdCreateAlignmentLayout) -> SettingsCmdRegressionGraphOption

"""


    SettingsCmdAlignmentTypeOption = None
    SettingsCmdCurveAndSpiralSettings = None
    SettingsCmdCurveTessellationOption = None
    SettingsCmdRegressionGraphOption = None


class SettingsCmdCreateAlignmentReference(SettingsAlignment):
    # no doc
    def Dispose(self):
        """ Dispose(self: DisposableWrapper, A_0: bool) """
        pass


class SettingsCmdCreateArcByBestFit(SettingsGeneral):
    # no doc
    def Dispose(self):
        """ Dispose(self: DisposableWrapper, A_0: bool) """
        pass

    CurveTessellationOption = property(lambda self: object(), lambda self, v: None, lambda self: None)  # default
    """Get: CurveTessellationOption(self: SettingsCmdCreateArcByBestFit) -> SettingsCmdCurveTessellationOption

"""

    RegressionGraphOption = property(lambda self: object(), lambda self, v: None, lambda self: None)  # default
    """Get: RegressionGraphOption(self: SettingsCmdCreateArcByBestFit) -> SettingsCmdRegressionGraphOption

"""


    SettingsCmdCurveTessellationOption = None
    SettingsCmdRegressionGraphOption = None


class SettingsCmdCreateAssembly(SettingsAssembly):
    # no doc
    def Dispose(self):
        """ Dispose(self: DisposableWrapper, A_0: bool) """
        pass


class SettingsCmdCreateAssemblyTool(SettingsAssembly):
    # no doc
    def Dispose(self):
        """ Dispose(self: DisposableWrapper, A_0: bool) """
        pass


class SettingsCmdCreateCantView(SettingsCantView):
    # no doc
    def Dispose(self):
        """ Dispose(self: DisposableWrapper, A_0: bool) """
        pass


class SettingsCmdCreateCatchmentFromObject(SettingsCatchment):
    # no doc
    def Dispose(self):
        """ Dispose(self: DisposableWrapper, A_0: bool) """
        pass

    Catchment = property(lambda self: object(), lambda self, v: None, lambda self: None)  # default
    """Get: Catchment(self: SettingsCmdCreateCatchmentFromObject) -> SettingsCmdCatchment

"""

    ChannelFlow = property(lambda self: object(), lambda self, v: None, lambda self: None)  # default
    """Get: ChannelFlow(self: SettingsCmdCreateCatchmentFromObject) -> SettingsCmdChannelFlow

"""

    HydrologicalProperties = property(lambda self: object(), lambda self, v: None, lambda self: None)  # default
    """Get: HydrologicalProperties(self: SettingsCmdCreateCatchmentFromObject) -> SettingsCmdHydrologicalProperties

"""

    ShallowConcentratedFlow = property(lambda self: object(), lambda self, v: None, lambda self: None)  # default
    """Get: ShallowConcentratedFlow(self: SettingsCmdCreateCatchmentFromObject) -> SettingsCmdShallowConcentratedFlow

"""

    SheetFlow = property(lambda self: object(), lambda self, v: None, lambda self: None)  # default
    """Get: SheetFlow(self: SettingsCmdCreateCatchmentFromObject) -> SettingsCmdSheetFlow

"""

    TimeOfConcentrationMethod = property(lambda self: object(), lambda self, v: None, lambda self: None)  # default
    """Get: TimeOfConcentrationMethod(self: SettingsCmdCreateCatchmentFromObject) -> PropertyEnum[CatchmentTimeOfConcentrationMethodType]

"""


    SettingsCmdCatchment = None
    SettingsCmdChannelFlow = None
    SettingsCmdHydrologicalProperties = None
    SettingsCmdShallowConcentratedFlow = None
    SettingsCmdSheetFlow = None


class SettingsCmdCreateCatchmentFromSurface(SettingsCatchment):
    # no doc
    def Dispose(self):
        """ Dispose(self: DisposableWrapper, A_0: bool) """
        pass

    Catchment = property(lambda self: object(), lambda self, v: None, lambda self: None)  # default
    """Get: Catchment(self: SettingsCmdCreateCatchmentFromSurface) -> SettingsCmdCatchment

"""

    ChannelFlow = property(lambda self: object(), lambda self, v: None, lambda self: None)  # default
    """Get: ChannelFlow(self: SettingsCmdCreateCatchmentFromSurface) -> SettingsCmdChannelFlow

"""

    HydrologicalProperties = property(lambda self: object(), lambda self, v: None, lambda self: None)  # default
    """Get: HydrologicalProperties(self: SettingsCmdCreateCatchmentFromSurface) -> SettingsCmdHydrologicalProperties

"""

    ShallowConcentratedFlow = property(lambda self: object(), lambda self, v: None, lambda self: None)  # default
    """Get: ShallowConcentratedFlow(self: SettingsCmdCreateCatchmentFromSurface) -> SettingsCmdShallowConcentratedFlow

"""

    SheetFlow = property(lambda self: object(), lambda self, v: None, lambda self: None)  # default
    """Get: SheetFlow(self: SettingsCmdCreateCatchmentFromSurface) -> SettingsCmdSheetFlow

"""

    TimeOfConcentrationMethod = property(lambda self: object(), lambda self, v: None, lambda self: None)  # default
    """Get: TimeOfConcentrationMethod(self: SettingsCmdCreateCatchmentFromSurface) -> PropertyEnum[CatchmentTimeOfConcentrationMethodType]

"""


    SettingsCmdCatchment = None
    SettingsCmdChannelFlow = None
    SettingsCmdHydrologicalProperties = None
    SettingsCmdShallowConcentratedFlow = None
    SettingsCmdSheetFlow = None


class SettingsCmdCreateCatchmentGroup(SettingsCatchment):
    # no doc
    def Dispose(self):
        """ Dispose(self: DisposableWrapper, A_0: bool) """
        pass


class SettingsCmdCreateCorridor(SettingsCorridor):
    # no doc
    def Dispose(self):
        """ Dispose(self: DisposableWrapper, A_0: bool) """
        pass

    AssemblyInsertion = property(lambda self: object(), lambda self, v: None, lambda self: None)  # default
    """Get: AssemblyInsertion(self: SettingsCmdCreateCorridor) -> SettingsCmdAssemblyInsertion

"""


    SettingsCmdAssemblyInsertion = None


class SettingsGrading(SettingsAmbient):
    # no doc
    def Dispose(self):
        """ Dispose(self: DisposableWrapper, A_0: bool) """
        pass

    NameFormat = property(lambda self: object(), lambda self, v: None, lambda self: None)  # default
    """Get: NameFormat(self: SettingsGrading) -> SettingsNameFormat

"""

    Styles = property(lambda self: object(), lambda self, v: None, lambda self: None)  # default
    """Get: Styles(self: SettingsGrading) -> SettingsStyles

"""


    SettingsNameFormat = None
    SettingsStyles = None


class SettingsCmdCreateFeatureLineFromAlign(SettingsGrading):
    # no doc
    def Dispose(self):
        """ Dispose(self: DisposableWrapper, A_0: bool) """
        pass

    FeatureLineCreation = property(lambda self: object(), lambda self, v: None, lambda self: None)  # default
    """Get: FeatureLineCreation(self: SettingsCmdCreateFeatureLineFromAlign) -> SettingsCmdFeatureLineCreation

"""


    SettingsCmdFeatureLineCreation = None


class SettingsCmdCreateFeatureLines(SettingsGrading):
    # no doc
    def Dispose(self):
        """ Dispose(self: DisposableWrapper, A_0: bool) """
        pass

    FeatureLineCreation = property(lambda self: object(), lambda self, v: None, lambda self: None)  # default
    """Get: FeatureLineCreation(self: SettingsCmdCreateFeatureLines) -> SettingsCmdFeatureLineCreation

"""


    SettingsCmdFeatureLineCreation = None


class SettingsCmdCreateFlowSegment(SettingsCatchment):
    # no doc
    def Dispose(self):
        """ Dispose(self: DisposableWrapper, A_0: bool) """
        pass

    ChannelFlow = property(lambda self: object(), lambda self, v: None, lambda self: None)  # default
    """Get: ChannelFlow(self: SettingsCmdCreateFlowSegment) -> SettingsCmdChannelFlow

"""

    ShallowConcentratedFlow = property(lambda self: object(), lambda self, v: None, lambda self: None)  # default
    """Get: ShallowConcentratedFlow(self: SettingsCmdCreateFlowSegment) -> SettingsCmdShallowConcentratedFlow

"""

    SheetFlow = property(lambda self: object(), lambda self, v: None, lambda self: None)  # default
    """Get: SheetFlow(self: SettingsCmdCreateFlowSegment) -> SettingsCmdSheetFlow

"""


    SettingsCmdChannelFlow = None
    SettingsCmdShallowConcentratedFlow = None
    SettingsCmdSheetFlow = None


class SettingsCmdCreateGrading(SettingsGrading):
    # no doc
    def Dispose(self):
        """ Dispose(self: DisposableWrapper, A_0: bool) """
        pass

    GradingCreation = property(lambda self: object(), lambda self, v: None, lambda self: None)  # default
    """Get: GradingCreation(self: SettingsCmdCreateGrading) -> SettingsCmdGradingCreation

"""


    SettingsCmdGradingCreation = None


class SettingsCmdCreateGradingGroup(SettingsGrading):
    # no doc
    def Dispose(self):
        """ Dispose(self: DisposableWrapper, A_0: bool) """
        pass

    GradingGroupCreation = property(lambda self: object(), lambda self, v: None, lambda self: None)  # default
    """Get: GradingGroupCreation(self: SettingsCmdCreateGradingGroup) -> SettingsCmdGradingGroupCreation

"""


    SettingsCmdGradingGroupCreation = None


class SettingsCmdCreateInterferenceCheck(SettingsPipeNetwork):
    # no doc
    def Dispose(self):
        """ Dispose(self: DisposableWrapper, A_0: bool) """
        pass

    InterferenceCriteria = property(lambda self: object(), lambda self, v: None, lambda self: None)  # default
    """Get: InterferenceCriteria(self: SettingsCmdCreateInterferenceCheck) -> SettingsCmdInterferenceCriteria

"""


    SettingsCmdInterferenceCriteria = None


class SettingsCmdCreateIntersection(SettingsIntersection):
    # no doc
    def Dispose(self):
        """ Dispose(self: DisposableWrapper, A_0: bool) """
        pass

    AssemblyInsertion = property(lambda self: object(), lambda self, v: None, lambda self: None)  # default
    """Get: AssemblyInsertion(self: SettingsCmdCreateIntersection) -> SettingsCmdAssemblyInsertion

"""

    CrossSlopes = property(lambda self: object(), lambda self, v: None, lambda self: None)  # default
    """Get: CrossSlopes(self: SettingsCmdCreateIntersection) -> SettingsCmdCrossSlopes

"""

    CurbReturnParameters = property(lambda self: object(), lambda self, v: None, lambda self: None)  # default
    """Get: CurbReturnParameters(self: SettingsCmdCreateIntersection) -> SettingsCmdCurbReturnParameters

"""

    CurbReturnProfileRules = property(lambda self: object(), lambda self, v: None, lambda self: None)  # default
    """Get: CurbReturnProfileRules(self: SettingsCmdCreateIntersection) -> SettingsCmdCurbReturnProfileRules

"""

    IntersectionOptions = property(lambda self: object(), lambda self, v: None, lambda self: None)  # default
    """Get: IntersectionOptions(self: SettingsCmdCreateIntersection) -> SettingsCmdIntersectionOptions

"""

    Offsets = property(lambda self: object(), lambda self, v: None, lambda self: None)  # default
    """Get: Offsets(self: SettingsCmdCreateIntersection) -> SettingsCmdOffsets

"""

    SecondaryRoadProfileRules = property(lambda self: object(), lambda self, v: None, lambda self: None)  # default
    """Get: SecondaryRoadProfileRules(self: SettingsCmdCreateIntersection) -> SettingsCmdSecondaryRoadProfileRules

"""

    WideningParameters = property(lambda self: object(), lambda self, v: None, lambda self: None)  # default
    """Get: WideningParameters(self: SettingsCmdCreateIntersection) -> SettingsCmdWideningParameters

"""


    SettingsCmdAssemblyInsertion = None
    SettingsCmdCrossSlopes = None
    SettingsCmdCurbReturnParameters = None
    SettingsCmdCurbReturnProfileRules = None
    SettingsCmdIntersectionOptions = None
    SettingsCmdOffsets = None
    SettingsCmdSecondaryRoadProfileRules = None
    SettingsCmdWideningParameters = None


class SettingsCmdCreateLineByBestFit(SettingsGeneral):
    # no doc
    def Dispose(self):
        """ Dispose(self: DisposableWrapper, A_0: bool) """
        pass

    CurveTessellationOption = property(lambda self: object(), lambda self, v: None, lambda self: None)  # default
    """Get: CurveTessellationOption(self: SettingsCmdCreateLineByBestFit) -> SettingsCmdCurveTessellationOption

"""

    RegressionGraphOption = property(lambda self: object(), lambda self, v: None, lambda self: None)  # default
    """Get: RegressionGraphOption(self: SettingsCmdCreateLineByBestFit) -> SettingsCmdRegressionGraphOption

"""


    SettingsCmdCurveTessellationOption = None
    SettingsCmdRegressionGraphOption = None


class SettingsMassHaulView(SettingsAmbient):
    # no doc
    def Dispose(self):
        """ Dispose(self: DisposableWrapper, A_0: bool) """
        pass

    MassHaulCreation = property(lambda self: object(), lambda self, v: None, lambda self: None)  # default
    """Get: MassHaulCreation(self: SettingsMassHaulView) -> SettingsMassHaulCreation

"""

    NameFormat = property(lambda self: object(), lambda self, v: None, lambda self: None)  # default
    """Get: NameFormat(self: SettingsMassHaulView) -> SettingsNameFormat

"""

    Styles = property(lambda self: object(), lambda self, v: None, lambda self: None)  # default
    """Get: Styles(self: SettingsMassHaulView) -> SettingsStyles

"""


    SettingsMassHaulCreation = None
    SettingsNameFormat = None
    SettingsStyles = None


class SettingsCmdCreateMassHaulDiagram(SettingsMassHaulView):
    # no doc
    def Dispose(self):
        """ Dispose(self: DisposableWrapper, A_0: bool) """
        pass

    MassHaulCreation = property(lambda self: object(), lambda self, v: None, lambda self: None)  # default
    """Get: MassHaulCreation(self: SettingsCmdCreateMassHaulDiagram) -> SettingsCmdMassHaulCreation

"""


    SettingsCmdMassHaulCreation = None


class SettingsCmdCreateMultipleProfileView(SettingsProfileView):
    # no doc
    def Dispose(self):
        """ Dispose(self: DisposableWrapper, A_0: bool) """
        pass

    MultipleProfileViewCreation = property(lambda self: object(), lambda self, v: None, lambda self: None)  # default
    """Get: MultipleProfileViewCreation(self: SettingsCmdCreateMultipleProfileView) -> SettingsCmdMultipleProfileViewCreation

"""


    SettingsCmdMultipleProfileViewCreation = None


class SettingsCmdCreateMultipleSectionView(SettingsSectionView):
    # no doc
    def Dispose(self):
        """ Dispose(self: DisposableWrapper, A_0: bool) """
        pass

    MultipleSectionViewCreation = property(lambda self: object(), lambda self, v: None, lambda self: None)  # default
    """Get: MultipleSectionViewCreation(self: SettingsCmdCreateMultipleSectionView) -> SettingsCmdMultipleSectionViewCreation

"""

    TableCreation = property(lambda self: object(), lambda self, v: None, lambda self: None)  # default
    """Get: TableCreation(self: SettingsCmdCreateMultipleSectionView) -> SettingsCmdTableCreation

"""


    SettingsCmdMultipleSectionViewCreation = None
    SettingsCmdTableCreation = None


class SettingsCmdCreateNetwork(SettingsPipeNetwork):
    # no doc
    def Dispose(self):
        """ Dispose(self: DisposableWrapper, A_0: bool) """
        pass

    DefaultLayoutCommand = property(lambda self: object(), lambda self, v: None, lambda self: None)  # default
    """Get: DefaultLayoutCommand(self: SettingsCmdCreateNetwork) -> PropertyEnum[NetworkDefaultLayoutCommandType]

"""

    LabelNewParts = property(lambda self: object(), lambda self, v: None, lambda self: None)  # default
    """Get: LabelNewParts(self: SettingsCmdCreateNetwork) -> SettingsCmdLabelNewParts

"""


    SettingsCmdLabelNewParts = None


class SettingsCmdCreateNetworkFromObject(SettingsPipeNetwork):
    # no doc
    def Dispose(self):
        """ Dispose(self: DisposableWrapper, A_0: bool) """
        pass


class SettingsCmdCreateNetworkPartsList(SettingsPipeNetwork):
    # no doc
    def Dispose(self):
        """ Dispose(self: DisposableWrapper, A_0: bool) """
        pass


class SettingsCmdCreateNetworkPartsListFull(SettingsPipeNetwork):
    # no doc
    def Dispose(self):
        """ Dispose(self: DisposableWrapper, A_0: bool) """
        pass


class SettingsCmdCreateNetworkReference(SettingsPipeNetwork):
    # no doc
    def Dispose(self):
        """ Dispose(self: DisposableWrapper, A_0: bool) """
        pass


class SettingsCmdCreateOffsetAlignment(SettingsAlignment):
    # no doc
    def Dispose(self):
        """ Dispose(self: DisposableWrapper, A_0: bool) """
        pass

    OffsetAlignmentOptions = property(lambda self: object(), lambda self, v: None, lambda self: None)  # default
    """Get: OffsetAlignmentOptions(self: SettingsCmdCreateOffsetAlignment) -> SettingsCmdOffsetAlignmentOptions

"""


    SettingsCmdOffsetAlignmentOptions = None


class SettingsCmdCreateParabolaByBestFit(SettingsGeneral):
    # no doc
    def Dispose(self):
        """ Dispose(self: DisposableWrapper, A_0: bool) """
        pass

    CurveTessellationOption = property(lambda self: object(), lambda self, v: None, lambda self: None)  # default
    """Get: CurveTessellationOption(self: SettingsCmdCreateParabolaByBestFit) -> SettingsCmdCurveTessellationOption

"""

    RegressionGraphOption = property(lambda self: object(), lambda self, v: None, lambda self: None)  # default
    """Get: RegressionGraphOption(self: SettingsCmdCreateParabolaByBestFit) -> SettingsCmdRegressionGraphOption

"""


    SettingsCmdCurveTessellationOption = None
    SettingsCmdRegressionGraphOption = None


class SettingsCmdCreateParcelByLayout(SettingsParcel):
    # no doc
    def Dispose(self):
        """ Dispose(self: DisposableWrapper, A_0: bool) """
        pass

    AutomaticLayout = property(lambda self: object(), lambda self, v: None, lambda self: None)  # default
    """Get: AutomaticLayout(self: SettingsCmdCreateParcelByLayout) -> SettingsCmdAutomaticLayout

"""

    ConvertFromEntities = property(lambda self: object(), lambda self, v: None, lambda self: None)  # default
    """Get: ConvertFromEntities(self: SettingsCmdCreateParcelByLayout) -> SettingsCmdConvertFromEntities

"""

    ParcelSizing = property(lambda self: object(), lambda self, v: None, lambda self: None)  # default
    """Get: ParcelSizing(self: SettingsCmdCreateParcelByLayout) -> SettingsCmdParcelSizing

"""

    PreviewGraphics = property(lambda self: object(), lambda self, v: None, lambda self: None)  # default
    """Get: PreviewGraphics(self: SettingsCmdCreateParcelByLayout) -> SettingsCmdPreviewGraphics

"""


    SettingsCmdAutomaticLayout = None
    SettingsCmdConvertFromEntities = None
    SettingsCmdParcelSizing = None
    SettingsCmdPreviewGraphics = None


class SettingsCmdCreateParcelFromObjects(SettingsParcel):
    # no doc
    def Dispose(self):
        """ Dispose(self: DisposableWrapper, A_0: bool) """
        pass

    ConvertFromEntities = property(lambda self: object(), lambda self, v: None, lambda self: None)  # default
    """Get: ConvertFromEntities(self: SettingsCmdCreateParcelFromObjects) -> SettingsCmdConvertFromEntities

"""


    SettingsCmdConvertFromEntities = None


class SettingsCmdCreateParcelROW(SettingsParcel):
    # no doc
    def Dispose(self):
        """ Dispose(self: DisposableWrapper, A_0: bool) """
        pass

    CleanupAtAlignmentIntersections = property(lambda self: object(), lambda self, v: None, lambda self: None)  # default
    """Get: CleanupAtAlignmentIntersections(self: SettingsCmdCreateParcelROW) -> SettingsCmdCleanupAtAlignmentIntersections

"""

    CleanupAtParcelBoundaries = property(lambda self: object(), lambda self, v: None, lambda self: None)  # default
    """Get: CleanupAtParcelBoundaries(self: SettingsCmdCreateParcelROW) -> SettingsCmdCleanupAtParcelBoundaries

"""

    CreateParcelRightOfWay = property(lambda self: object(), lambda self, v: None, lambda self: None)  # default
    """Get: CreateParcelRightOfWay(self: SettingsCmdCreateParcelROW) -> SettingsCmdCreateParcelRightOfWay

"""


    SettingsCmdCleanupAtAlignmentIntersections = None
    SettingsCmdCleanupAtParcelBoundaries = None
    SettingsCmdCreateParcelRightOfWay = None


class SettingsCmdCreatePointCloud(SettingsPointCloud):
    # no doc
    def Dispose(self):
        """ Dispose(self: DisposableWrapper, A_0: bool) """
        pass

    DefaultLayer = property(lambda self: object(), lambda self, v: None, lambda self: None)  # default
    """Get: DefaultLayer(self: SettingsCmdCreatePointCloud) -> SettingsCmdDefaultLayer

"""

    FileFormat = property(lambda self: object(), lambda self, v: None, lambda self: None)  # default
    """Get: FileFormat(self: SettingsCmdCreatePointCloud) -> PropertyEnum[PointCloudDefaultFileExtensionType]

"""


    SettingsCmdDefaultLayer = None


class SettingsCmdCreatePointGroup(SettingsPoint):
    # no doc
    def Dispose(self):
        """ Dispose(self: DisposableWrapper, A_0: bool) """
        pass


class SettingsCmdCreatePoints(SettingsPoint):
    # no doc
    def Dispose(self):
        """ Dispose(self: DisposableWrapper, A_0: bool) """
        pass

    Layer = property(lambda self: object(), lambda self, v: None, lambda self: None)  # default
    """Get: Layer(self: SettingsCmdCreatePoints) -> SettingsCmdLayer

"""

    PointIdentity = property(lambda self: object(), lambda self, v: None, lambda self: None)  # default
    """Get: PointIdentity(self: SettingsCmdCreatePoints) -> SettingsCmdPointIdentity

"""

    PointsCreation = property(lambda self: object(), lambda self, v: None, lambda self: None)  # default
    """Get: PointsCreation(self: SettingsCmdCreatePoints) -> SettingsCmdPointsCreation

"""


    SettingsCmdLayer = None
    SettingsCmdPointIdentity = None
    SettingsCmdPointsCreation = None


class SettingsCmdCreatePointsFromCorridor(SettingsCorridor):
    # no doc
    def Dispose(self):
        """ Dispose(self: DisposableWrapper, A_0: bool) """
        pass


class SettingsCmdCreatePolylineFromCorridor(SettingsCorridor):
    # no doc
    def Dispose(self):
        """ Dispose(self: DisposableWrapper, A_0: bool) """
        pass


class SettingsSuperelevationView(SettingsAmbient):
    # no doc
    def Dispose(self):
        """ Dispose(self: DisposableWrapper, A_0: bool) """
        pass

    NameFormat = property(lambda self: object(), lambda self, v: None, lambda self: None)  # default
    """Get: NameFormat(self: SettingsSuperelevationView) -> SettingsNameFormat

"""

    Styles = property(lambda self: object(), lambda self, v: None, lambda self: None)  # default
    """Get: Styles(self: SettingsSuperelevationView) -> SettingsStyles

"""


    SettingsNameFormat = None
    SettingsStyles = None


class SettingsCmdCreatePolylineFromSuper(SettingsSuperelevationView):
    # no doc
    def Dispose(self):
        """ Dispose(self: DisposableWrapper, A_0: bool) """
        pass


class SettingsCmdCreatePressureFromIndModel(SettingsPressureNetwork):
    # no doc
    def Dispose(self):
        """ Dispose(self: DisposableWrapper, A_0: bool) """
        pass


class SettingsCmdCreatePressureNetwork(SettingsPressureNetwork):
    # no doc
    def Dispose(self):
        """ Dispose(self: DisposableWrapper, A_0: bool) """
        pass

    DepthOfCover = property(lambda self: object(), lambda self, v: None, lambda self: None)  # default
    """Get: DepthOfCover(self: SettingsCmdCreatePressureNetwork) -> SettingsCmdDepthOfCover

"""

    LabelNewParts = property(lambda self: object(), lambda self, v: None, lambda self: None)  # default
    """Get: LabelNewParts(self: SettingsCmdCreatePressureNetwork) -> SettingsCmdLabelNewParts

"""


    SettingsCmdDepthOfCover = None
    SettingsCmdLabelNewParts = None


class SettingsCmdCreatePressurePartList(SettingsPressureNetwork):
    # no doc
    def Dispose(self):
        """ Dispose(self: DisposableWrapper, A_0: bool) """
        pass


class SettingsCmdCreatePressurePartListFull(SettingsPressureNetwork):
    # no doc
    def Dispose(self):
        """ Dispose(self: DisposableWrapper, A_0: bool) """
        pass


class SettingsCmdCreateProfileFromCorridor(SettingsCorridor):
    # no doc
    def Dispose(self):
        """ Dispose(self: DisposableWrapper, A_0: bool) """
        pass

    CriteriaBasedDesignOptions = property(lambda self: object(), lambda self, v: None, lambda self: None)  # default
    """Get: CriteriaBasedDesignOptions(self: SettingsCmdCreateProfileFromCorridor) -> SettingsCmdCriteriaBasedDesignOptions

"""


    SettingsCmdCriteriaBasedDesignOptions = None


class SettingsProfile(SettingsAmbient):
    # no doc
    def Dispose(self):
        """ Dispose(self: DisposableWrapper, A_0: bool) """
        pass

    CriteriaBasedDesignOptions = property(lambda self: object(), lambda self, v: None, lambda self: None)  # default
    """Get: CriteriaBasedDesignOptions(self: SettingsProfile) -> SettingsCriteriaBasedDesignOptions

"""

    DefaultNameFormat = property(lambda self: object(), lambda self, v: None, lambda self: None)  # default
    """Get: DefaultNameFormat(self: SettingsProfile) -> SettingsDefaultNameFormat

"""

    ProfilesCreation = property(lambda self: object(), lambda self, v: None, lambda self: None)  # default
    """Get: ProfilesCreation(self: SettingsProfile) -> SettingsProfileCreation

"""

    StyleSettings = property(lambda self: object(), lambda self, v: None, lambda self: None)  # default
    """Get: StyleSettings(self: SettingsProfile) -> SettingsStyles

"""


    SettingsCriteriaBasedDesignOptions = None
    SettingsDefaultNameFormat = None
    SettingsProfileCreation = None
    SettingsStyles = None


class SettingsCmdCreateProfileFromFile(SettingsProfile):
    # no doc
    def Dispose(self):
        """ Dispose(self: DisposableWrapper, A_0: bool) """
        pass


class SettingsCmdCreateProfileFromSurface(SettingsProfile):
    # no doc
    def Dispose(self):
        """ Dispose(self: DisposableWrapper, A_0: bool) """
        pass

    Geometry = property(lambda self: object(), lambda self, v: None, lambda self: None)  # default
    """Get: Geometry(self: SettingsCmdCreateProfileFromSurface) -> SettingsCmdGeometry

"""


    SettingsCmdGeometry = None


class SettingsCmdCreateProfileLayout(SettingsProfile):
    # no doc
    def Dispose(self):
        """ Dispose(self: DisposableWrapper, A_0: bool) """
        pass

    CurveTessellationOption = property(lambda self: object(), lambda self, v: None, lambda self: None)  # default
    """Get: CurveTessellationOption(self: SettingsCmdCreateProfileLayout) -> SettingsCmdCurveTessellationOption

"""

    RegressionGraphOption = property(lambda self: object(), lambda self, v: None, lambda self: None)  # default
    """Get: RegressionGraphOption(self: SettingsCmdCreateProfileLayout) -> SettingsCmdRegressionGraphOption

"""


    SettingsCmdCurveTessellationOption = None
    SettingsCmdRegressionGraphOption = None


class SettingsCmdCreateProfileReference(SettingsProfile):
    # no doc
    def Dispose(self):
        """ Dispose(self: DisposableWrapper, A_0: bool) """
        pass


class SettingsCmdCreateProfileView(SettingsProfileView):
    # no doc
    def Dispose(self):
        """ Dispose(self: DisposableWrapper, A_0: bool) """
        pass


class SettingsCmdCreateQuickProfile(SettingsProfile):
    # no doc
    def Dispose(self):
        """ Dispose(self: DisposableWrapper, A_0: bool) """
        pass

    QuickProfile = property(lambda self: object(), lambda self, v: None, lambda self: None)  # default
    """Get: QuickProfile(self: SettingsCmdCreateQuickProfile) -> SettingsCmdQuickProfile

"""


    SettingsCmdQuickProfile = None


class SettingsSampleLine(SettingsAmbient):
    # no doc
    def Dispose(self):
        """ Dispose(self: DisposableWrapper, A_0: bool) """
        pass

    NameFormat = property(lambda self: object(), lambda self, v: None, lambda self: None)  # default
    """Get: NameFormat(self: SettingsSampleLine) -> SettingsNameFormat

"""

    Styles = property(lambda self: object(), lambda self, v: None, lambda self: None)  # default
    """Get: Styles(self: SettingsSampleLine) -> SettingsStyles

"""


    SettingsNameFormat = None
    SettingsStyles = None


class SettingsCmdCreateSampleLines(SettingsSampleLine):
    # no doc
    def Dispose(self):
        """ Dispose(self: DisposableWrapper, A_0: bool) """
        pass

    AdditionalSampleControls = property(lambda self: object(), lambda self, v: None, lambda self: None)  # default
    """Get: AdditionalSampleControls(self: SettingsCmdCreateSampleLines) -> SettingsCmdAdditionalSampleControls

"""

    Miscellaneous = property(lambda self: object(), lambda self, v: None, lambda self: None)  # default
    """Get: Miscellaneous(self: SettingsCmdCreateSampleLines) -> SettingsCmdMiscellaneous

"""

    SamplingIncrements = property(lambda self: object(), lambda self, v: None, lambda self: None)  # default
    """Get: SamplingIncrements(self: SettingsCmdCreateSampleLines) -> SettingsCmdSamplingIncrements

"""

    SwathWidths = property(lambda self: object(), lambda self, v: None, lambda self: None)  # default
    """Get: SwathWidths(self: SettingsCmdCreateSampleLines) -> SettingsCmdSwathWidths

"""


    SettingsCmdAdditionalSampleControls = None
    SettingsCmdMiscellaneous = None
    SettingsCmdSamplingIncrements = None
    SettingsCmdSwathWidths = None


class SettingsCmdCreateSectionSheets(SettingsSectionView):
    # no doc
    def Dispose(self):
        """ Dispose(self: DisposableWrapper, A_0: bool) """
        pass

    SheetCreation = property(lambda self: object(), lambda self, v: None, lambda self: None)  # default
    """Get: SheetCreation(self: SettingsCmdCreateSectionSheets) -> SettingsCmdSheetCreation

"""


    SettingsCmdSheetCreation = None


class SettingsCmdCreateSectionView(SettingsSectionView):
    # no doc
    def Dispose(self):
        """ Dispose(self: DisposableWrapper, A_0: bool) """
        pass

    TableCreation = property(lambda self: object(), lambda self, v: None, lambda self: None)  # default
    """Get: TableCreation(self: SettingsCmdCreateSectionView) -> SettingsCmdTableCreation

"""


    SettingsCmdTableCreation = None


class SettingsViewFrameGroup(SettingsAmbient):
    # no doc
    def Dispose(self):
        """ Dispose(self: DisposableWrapper, A_0: bool) """
        pass

    Information = property(lambda self: object(), lambda self, v: None, lambda self: None)  # default
    """Get: Information(self: SettingsViewFrameGroup) -> SettingsInformation

"""

    NameFormat = property(lambda self: object(), lambda self, v: None, lambda self: None)  # default
    """Get: NameFormat(self: SettingsViewFrameGroup) -> SettingsNameFormat

"""

    Styles = property(lambda self: object(), lambda self, v: None, lambda self: None)  # default
    """Get: Styles(self: SettingsViewFrameGroup) -> SettingsStyles

"""


    SettingsInformation = None
    SettingsNameFormat = None
    SettingsStyles = None


class SettingsCmdCreateSheets(SettingsViewFrameGroup):
    # no doc
    def Dispose(self):
        """ Dispose(self: DisposableWrapper, A_0: bool) """
        pass

    SheetCreation = property(lambda self: object(), lambda self, v: None, lambda self: None)  # default
    """Get: SheetCreation(self: SettingsCmdCreateSheets) -> SettingsCmdSheetCreation

"""


    SettingsCmdSheetCreation = None


class SettingsCmdCreateSimpleCorridor(SettingsCorridor):
    # no doc
    def Dispose(self):
        """ Dispose(self: DisposableWrapper, A_0: bool) """
        pass

    AssemblyInsertion = property(lambda self: object(), lambda self, v: None, lambda self: None)  # default
    """Get: AssemblyInsertion(self: SettingsCmdCreateSimpleCorridor) -> SettingsCmdAssemblyInsertion

"""


    SettingsCmdAssemblyInsertion = None


class SettingsCmdCreateSite(SettingsParcel):
    # no doc
    def Dispose(self):
        """ Dispose(self: DisposableWrapper, A_0: bool) """
        pass

    Alignment = property(lambda self: object(), lambda self, v: None, lambda self: None)  # default
    """Get: Alignment(self: SettingsCmdCreateSite) -> SettingsCmdAlignment

"""

    FeatureLine = property(lambda self: object(), lambda self, v: None, lambda self: None)  # default
    """Get: FeatureLine(self: SettingsCmdCreateSite) -> SettingsCmdFeatureLine

"""

    Parcel = property(lambda self: object(), lambda self, v: None, lambda self: None)  # default
    """Get: Parcel(self: SettingsCmdCreateSite) -> SettingsCmdParcel

"""


    SettingsCmdAlignment = None
    SettingsCmdFeatureLine = None
    SettingsCmdParcel = None


class SettingsSubassembly(SettingsAmbient):
    # no doc
    def Dispose(self):
        """ Dispose(self: DisposableWrapper, A_0: bool) """
        pass

    DefaultStyles = property(lambda self: object(), lambda self, v: None, lambda self: None)  # default
    """Get: DefaultStyles(self: SettingsSubassembly) -> SettingsDefaultStyles

"""

    NameFormat = property(lambda self: object(), lambda self, v: None, lambda self: None)  # default
    """Get: NameFormat(self: SettingsSubassembly) -> SettingsNameFormat

"""


    SettingsDefaultStyles = None
    SettingsNameFormat = None


class SettingsCmdCreateSubassemblyTool(SettingsSubassembly):
    # no doc
    def Dispose(self):
        """ Dispose(self: DisposableWrapper, A_0: bool) """
        pass

    SubassemblyOptions = property(lambda self: object(), lambda self, v: None, lambda self: None)  # default
    """Get: SubassemblyOptions(self: SettingsCmdCreateSubassemblyTool) -> SettingsCmdSubassemblyOptions

"""


    SettingsCmdSubassemblyOptions = None


class SettingsCmdCreateSubFromPline(SettingsSubassembly):
    # no doc
    def Dispose(self):
        """ Dispose(self: DisposableWrapper, A_0: bool) """
        pass

    CreateFromEntities = property(lambda self: object(), lambda self, v: None, lambda self: None)  # default
    """Get: CreateFromEntities(self: SettingsCmdCreateSubFromPline) -> SettingsCmdCreateFromEntities

"""


    SettingsCmdCreateFromEntities = None


class SettingsCmdCreateSuperelevationView(SettingsSuperelevationView):
    # no doc
    def Dispose(self):
        """ Dispose(self: DisposableWrapper, A_0: bool) """
        pass


class SettingsCmdCreateSurface(SettingsSurface):
    # no doc
    def Dispose(self):
        """ Dispose(self: DisposableWrapper, A_0: bool) """
        pass

    BuildOptions = property(lambda self: object(), lambda self, v: None, lambda self: None)  # default
    """Get: BuildOptions(self: SettingsCmdCreateSurface) -> SettingsCmdBuildOptions

"""

    NameFormat = property(lambda self: object(), lambda self, v: None, lambda self: None)  # default
    """Get: NameFormat(self: SettingsCmdCreateSurface) -> SettingsNameFormat

"""

    SurfaceCreation = property(lambda self: object(), lambda self, v: None, lambda self: None)  # default
    """Get: SurfaceCreation(self: SettingsCmdCreateSurface) -> SettingsCmdSurfaceCreation

"""


    SettingsCmdBuildOptions = None
    SettingsCmdSurfaceCreation = None
    SettingsNameFormat = None


class SettingsCmdCreateSurfaceFromTIN(SettingsSurface):
    # no doc
    def Dispose(self):
        """ Dispose(self: DisposableWrapper, A_0: bool) """
        pass


class SettingsCmdCreateSurfaceGridFromDEM(SettingsSurface):
    # no doc
    def Dispose(self):
        """ Dispose(self: DisposableWrapper, A_0: bool) """
        pass

    BuildOptions = property(lambda self: object(), lambda self, v: None, lambda self: None)  # default
    """Get: BuildOptions(self: SettingsCmdCreateSurfaceGridFromDEM) -> SettingsCmdBuildOptions

"""

    ImportOptions = property(lambda self: object(), lambda self, v: None, lambda self: None)  # default
    """Get: ImportOptions(self: SettingsCmdCreateSurfaceGridFromDEM) -> SettingsCmdImportOptions

"""


    SettingsCmdBuildOptions = None
    SettingsCmdImportOptions = None


class SettingsCmdCreateSurfaceReference(SettingsSurface):
    # no doc
    def Dispose(self):
        """ Dispose(self: DisposableWrapper, A_0: bool) """
        pass

    NameFormat = property(lambda self: object(), lambda self, v: None, lambda self: None)  # default
    """Get: NameFormat(self: SettingsCmdCreateSurfaceReference) -> SettingsNameFormat

"""


    SettingsNameFormat = None


class SettingsCmdCreateSurfaceWaterdrop(SettingsSurface):
    # no doc
    def Dispose(self):
        """ Dispose(self: DisposableWrapper, A_0: bool) """
        pass

    WaterdropMarker = property(lambda self: object(), lambda self, v: None, lambda self: None)  # default
    """Get: WaterdropMarker(self: SettingsCmdCreateSurfaceWaterdrop) -> SettingsCmdWaterdropMarker

"""

    WaterdropPath = property(lambda self: object(), lambda self, v: None, lambda self: None)  # default
    """Get: WaterdropPath(self: SettingsCmdCreateSurfaceWaterdrop) -> SettingsCmdWaterdropPath

"""


    SettingsCmdWaterdropMarker = None
    SettingsCmdWaterdropPath = None


class SettingsCmdCreateViewFrames(SettingsViewFrameGroup):
    # no doc
    def Dispose(self):
        """ Dispose(self: DisposableWrapper, A_0: bool) """
        pass

    ViewFrameCreation = property(lambda self: object(), lambda self, v: None, lambda self: None)  # default
    """Get: ViewFrameCreation(self: SettingsCmdCreateViewFrames) -> SettingsCmdViewFrameCreation

"""


    SettingsCmdViewFrameCreation = None


class SettingsCmdDrawFeatureLine(SettingsGrading):
    # no doc
    def Dispose(self):
        """ Dispose(self: DisposableWrapper, A_0: bool) """
        pass

    FeatureLineCreation = property(lambda self: object(), lambda self, v: None, lambda self: None)  # default
    """Get: FeatureLineCreation(self: SettingsCmdDrawFeatureLine) -> SettingsCmdFeatureLineCreation

"""


    SettingsCmdFeatureLineCreation = None


class SettingsCmdEditFlowSegments(SettingsCatchment):
    # no doc
    def Dispose(self):
        """ Dispose(self: DisposableWrapper, A_0: bool) """
        pass

    ChannelFlow = property(lambda self: object(), lambda self, v: None, lambda self: None)  # default
    """Get: ChannelFlow(self: SettingsCmdEditFlowSegments) -> SettingsCmdChannelFlow

"""

    ShallowConcentratedFlow = property(lambda self: object(), lambda self, v: None, lambda self: None)  # default
    """Get: ShallowConcentratedFlow(self: SettingsCmdEditFlowSegments) -> SettingsCmdShallowConcentratedFlow

"""

    SheetFlow = property(lambda self: object(), lambda self, v: None, lambda self: None)  # default
    """Get: SheetFlow(self: SettingsCmdEditFlowSegments) -> SettingsCmdSheetFlow

"""


    SettingsCmdChannelFlow = None
    SettingsCmdShallowConcentratedFlow = None
    SettingsCmdSheetFlow = None


class SettingsCmdEditInStormSewers(SettingsPipeNetwork):
    # no doc
    def Dispose(self):
        """ Dispose(self: DisposableWrapper, A_0: bool) """
        pass


class SettingsCmdEditSVGroupStyle(SettingsSectionView):
    # no doc
    def Dispose(self):
        """ Dispose(self: DisposableWrapper, A_0: bool) """
        pass


class SettingsCmdExportParcelAnalysis(SettingsParcel):
    # no doc
    def Dispose(self):
        """ Dispose(self: DisposableWrapper, A_0: bool) """
        pass

    ParcelAnalysis = property(lambda self: object(), lambda self, v: None, lambda self: None)  # default
    """Get: ParcelAnalysis(self: SettingsCmdExportParcelAnalysis) -> SettingsCmdParcelAnalysis

"""


    SettingsCmdParcelAnalysis = None


class SettingsCmdExportStormSewerData(SettingsPipeNetwork):
    # no doc
    def Dispose(self):
        """ Dispose(self: DisposableWrapper, A_0: bool) """
        pass


class SettingsCmdFeatureLinesFromCorridor(SettingsCorridor):
    # no doc
    def Dispose(self):
        """ Dispose(self: DisposableWrapper, A_0: bool) """
        pass

    FeatureLineCreation = property(lambda self: object(), lambda self, v: None, lambda self: None)  # default
    """Get: FeatureLineCreation(self: SettingsCmdFeatureLinesFromCorridor) -> SettingsCmdFeatureLineCreation

"""


    SettingsCmdFeatureLineCreation = None


class SettingsCmdFitCurveFeature(SettingsGrading):
    # no doc
    def Dispose(self):
        """ Dispose(self: DisposableWrapper, A_0: bool) """
        pass

    FeatureLineFitCurve = property(lambda self: object(), lambda self, v: None, lambda self: None)  # default
    """Get: FeatureLineFitCurve(self: SettingsCmdFitCurveFeature) -> SettingsCmdFeatureLineFitCurve

"""


    SettingsCmdFeatureLineFitCurve = None


class SettingsCmdGenerateQuantitiesReport(SettingsQuantityTakeoff):
    # no doc
    def Dispose(self):
        """ Dispose(self: DisposableWrapper, A_0: bool) """
        pass

    DisplayXmlReport = property(lambda self: object(), lambda self, v: None, lambda self: None)  # default
    """Get: DisplayXmlReport(self: SettingsCmdGenerateQuantitiesReport) -> PropertyBoolean

"""



class SettingsCmdGradingElevEditor(SettingsGrading):
    # no doc
    def Dispose(self):
        """ Dispose(self: DisposableWrapper, A_0: bool) """
        pass

    GradingElevationEditor = property(lambda self: object(), lambda self, v: None, lambda self: None)  # default
    """Get: GradingElevationEditor(self: SettingsCmdGradingElevEditor) -> SettingsCmdGradingElevationEditor

"""


    SettingsCmdGradingElevationEditor = None


class SettingsCmdGradingTools(SettingsGrading):
    # no doc
    def Dispose(self):
        """ Dispose(self: DisposableWrapper, A_0: bool) """
        pass

    GradingLayoutTools = property(lambda self: object(), lambda self, v: None, lambda self: None)  # default
    """Get: GradingLayoutTools(self: SettingsCmdGradingTools) -> SettingsCmdGradingLayoutTools

"""


    SettingsCmdGradingLayoutTools = None


class SettingsCmdGradingVolumeTools(SettingsGrading):
    # no doc
    def Dispose(self):
        """ Dispose(self: DisposableWrapper, A_0: bool) """
        pass

    LimitFeatureSelectionToCurrentGroup = property(lambda self: object(), lambda self, v: None, lambda self: None)  # default
    """Get: LimitFeatureSelectionToCurrentGroup(self: SettingsCmdGradingVolumeTools) -> PropertyBoolean

"""

    RaiseLowerElevationIncrement = property(lambda self: object(), lambda self, v: None, lambda self: None)  # default
    """Get: RaiseLowerElevationIncrement(self: SettingsCmdGradingVolumeTools) -> PropertyDouble

"""



class SettingsCmdImportBuildingSite(SettingsBuildingSite):
    # no doc
    def Dispose(self):
        """ Dispose(self: DisposableWrapper, A_0: bool) """
        pass


class SettingsCmdImportGISData(SettingsGeneral):
    # no doc
    def Dispose(self):
        """ Dispose(self: DisposableWrapper, A_0: bool) """
        pass

    PipeNetwork = property(lambda self: object(), lambda self, v: None, lambda self: None)  # default
    """Get: PipeNetwork(self: SettingsCmdImportGISData) -> SettingsCmdPipeNetwork

"""


    SettingsCmdPipeNetwork = None


class SettingsCmdImportStormSewerData(SettingsPipeNetwork):
    # no doc
    def Dispose(self):
        """ Dispose(self: DisposableWrapper, A_0: bool) """
        pass


class SettingsCmdJoinFeatures(SettingsGrading):
    # no doc
    def Dispose(self):
        """ Dispose(self: DisposableWrapper, A_0: bool) """
        pass

    FeatureLineJoin = property(lambda self: object(), lambda self, v: None, lambda self: None)  # default
    """Get: FeatureLineJoin(self: SettingsCmdJoinFeatures) -> SettingsCmdFeatureLineJoin

"""


    SettingsCmdFeatureLineJoin = None


class SettingsCmdLayoutSectionViewGroup(SettingsSectionView):
    # no doc
    def Dispose(self):
        """ Dispose(self: DisposableWrapper, A_0: bool) """
        pass


class SettingsCmdMapCheck(SettingsGeneral):
    # no doc
    def Dispose(self):
        """ Dispose(self: DisposableWrapper, A_0: bool) """
        pass

    Mapcheck = property(lambda self: object(), lambda self, v: None, lambda self: None)  # default
    """Get: Mapcheck(self: SettingsCmdMapCheck) -> SettingsCmdMapcheck

"""


    SettingsCmdMapcheck = None


class SettingsCmdMinimizeSurfaceFlatAreas(SettingsSurface):
    # no doc
    def Dispose(self):
        """ Dispose(self: DisposableWrapper, A_0: bool) """
        pass

    AddPointsToFlatEdges = property(lambda self: object(), lambda self, v: None, lambda self: None)  # default
    """Get: AddPointsToFlatEdges(self: SettingsCmdMinimizeSurfaceFlatAreas) -> PropertyBoolean

"""

    AddPointsToFlatTriangles = property(lambda self: object(), lambda self, v: None, lambda self: None)  # default
    """Get: AddPointsToFlatTriangles(self: SettingsCmdMinimizeSurfaceFlatAreas) -> PropertyBoolean

"""

    FillGapsInContour = property(lambda self: object(), lambda self, v: None, lambda self: None)  # default
    """Get: FillGapsInContour(self: SettingsCmdMinimizeSurfaceFlatAreas) -> PropertyBoolean

"""

    SwapEdges = property(lambda self: object(), lambda self, v: None, lambda self: None)  # default
    """Get: SwapEdges(self: SettingsCmdMinimizeSurfaceFlatAreas) -> PropertyBoolean

"""



class SettingsCmdMoveBlockstoAttribElev(SettingsSurface):
    # no doc
    def Dispose(self):
        """ Dispose(self: DisposableWrapper, A_0: bool) """
        pass


class SettingsCmdMoveBlocksToSurface(SettingsSurface):
    # no doc
    def Dispose(self):
        """ Dispose(self: DisposableWrapper, A_0: bool) """
        pass


class SettingsCmdMoveTextToElevation(SettingsSurface):
    # no doc
    def Dispose(self):
        """ Dispose(self: DisposableWrapper, A_0: bool) """
        pass


class SettingsCmdProjectObjectsToMultiSect(SettingsSectionView):
    # no doc
    def Dispose(self):
        """ Dispose(self: DisposableWrapper, A_0: bool) """
        pass

    ObjectSelectionOptions = property(lambda self: object(), lambda self, v: None, lambda self: None)  # default
    """Get: ObjectSelectionOptions(self: SettingsCmdProjectObjectsToMultiSect) -> SettingsCmdObjectSelectionOptions

"""


    SettingsCmdObjectSelectionOptions = None


class SettingsCmdProjectObjectsToProf(SettingsProfileView):
    # no doc
    def Dispose(self):
        """ Dispose(self: DisposableWrapper, A_0: bool) """
        pass


class SettingsCmdProjectObjectsToSect(SettingsSectionView):
    # no doc
    def Dispose(self):
        """ Dispose(self: DisposableWrapper, A_0: bool) """
        pass


class SettingsCmdReAddParcelAreaLabel(SettingsParcel):
    # no doc
    def Dispose(self):
        """ Dispose(self: DisposableWrapper, A_0: bool) """
        pass


class SettingsCmdReAddParcelSegmentLabels(SettingsParcel):
    # no doc
    def Dispose(self):
        """ Dispose(self: DisposableWrapper, A_0: bool) """
        pass


class SettingsCmdRenamePipeNetworkParts(SettingsPipeNetwork):
    # no doc
    def Dispose(self):
        """ Dispose(self: DisposableWrapper, A_0: bool) """
        pass


class SettingsCmdResetAnchorPipe(SettingsPipeNetwork):
    # no doc
    def Dispose(self):
        """ Dispose(self: DisposableWrapper, A_0: bool) """
        pass


class SettingsCmdReverseAlignmentDirection(SettingsAlignment):
    # no doc
    def Dispose(self):
        """ Dispose(self: DisposableWrapper, A_0: bool) """
        pass


class SettingsCmdRunDepthCheck(SettingsPressureNetwork):
    # no doc
    def Dispose(self):
        """ Dispose(self: DisposableWrapper, A_0: bool) """
        pass

    DepthCheckOption = property(lambda self: object(), lambda self, v: None, lambda self: None)  # default
    """Get: DepthCheckOption(self: SettingsCmdRunDepthCheck) -> SettingsCmdDepthCheckOption

"""


    SettingsCmdDepthCheckOption = None


class SettingsCmdRunDesignCheck(SettingsPressureNetwork):
    # no doc
    def Dispose(self):
        """ Dispose(self: DisposableWrapper, A_0: bool) """
        pass

    DesignCheckOption = property(lambda self: object(), lambda self, v: None, lambda self: None)  # default
    """Get: DesignCheckOption(self: SettingsCmdRunDesignCheck) -> SettingsCmdDesignCheckOption

"""


    SettingsCmdDesignCheckOption = None


class SettingsCmdShowGeodeticCalculator(SettingsPoint):
    # no doc
    def Dispose(self):
        """ Dispose(self: DisposableWrapper, A_0: bool) """
        pass


class SettingsCmdShowPointGroupProperties(SettingsPoint):
    # no doc
    def Dispose(self):
        """ Dispose(self: DisposableWrapper, A_0: bool) """
        pass


class SettingsCmdShowSpanningPipes(SettingsPipeNetwork):
    # no doc
    def Dispose(self):
        """ Dispose(self: DisposableWrapper, A_0: bool) """
        pass


class SettingsCmdSimplifySurface(SettingsSurface):
    # no doc
    def Dispose(self):
        """ Dispose(self: DisposableWrapper, A_0: bool) """
        pass

    MaximumChangeInElevation = property(lambda self: object(), lambda self, v: None, lambda self: None)  # default
    """Get: MaximumChangeInElevation(self: SettingsCmdSimplifySurface) -> PropertyDouble

"""

    PercentageOfPointsToRemove = property(lambda self: object(), lambda self, v: None, lambda self: None)  # default
    """Get: PercentageOfPointsToRemove(self: SettingsCmdSimplifySurface) -> PropertyDouble

"""

    RegionOptions = property(lambda self: object(), lambda self, v: None, lambda self: None)  # default
    """Get: RegionOptions(self: SettingsCmdSimplifySurface) -> PropertyEnum[SurfaceRegionOptionsType]

"""

    SimplifyMethod = property(lambda self: object(), lambda self, v: None, lambda self: None)  # default
    """Get: SimplifyMethod(self: SettingsCmdSimplifySurface) -> PropertyEnum[SurfaceSimplifyType]

"""

    UseMaximumChangeInElevation = property(lambda self: object(), lambda self, v: None, lambda self: None)  # default
    """Get: UseMaximumChangeInElevation(self: SettingsCmdSimplifySurface) -> PropertyBoolean

"""

    UsePercentageOfPointsToRemove = property(lambda self: object(), lambda self, v: None, lambda self: None)  # default
    """Get: UsePercentageOfPointsToRemove(self: SettingsCmdSimplifySurface) -> PropertyBoolean

"""



class SettingsCmdSuperimposeProfile(SettingsProfile):
    # no doc
    def Dispose(self):
        """ Dispose(self: DisposableWrapper, A_0: bool) """
        pass

    SuperimposeProfile = property(lambda self: object(), lambda self, v: None, lambda self: None)  # default
    """Get: SuperimposeProfile(self: SettingsCmdSuperimposeProfile) -> SettingsCmdSuperimposeProfileOption

"""


    SettingsCmdSuperimposeProfileOption = None


class SettingsCmdSurfaceExportToDem(SettingsSurface):
    # no doc
    def Dispose(self):
        """ Dispose(self: DisposableWrapper, A_0: bool) """
        pass

    ExportOptions = property(lambda self: object(), lambda self, v: None, lambda self: None)  # default
    """Get: ExportOptions(self: SettingsCmdSurfaceExportToDem) -> SettingsCmdExportOptions

"""


    SettingsCmdExportOptions = None


class SettingsCmdSurfaceExtractObjects(SettingsSurface):
    # no doc
    def Dispose(self):
        """ Dispose(self: DisposableWrapper, A_0: bool) """
        pass


class SettingsCmdTakeOff(SettingsQuantityTakeoff):
    # no doc
    def Dispose(self):
        """ Dispose(self: DisposableWrapper, A_0: bool) """
        pass

    ComputeTakeOffOption = property(lambda self: object(), lambda self, v: None, lambda self: None)  # default
    """Get: ComputeTakeOffOption(self: SettingsCmdTakeOff) -> SettingsCmdComputeTakeOff

"""


    SettingsCmdComputeTakeOff = None


class SettingsCmdViewEditCorridorSection(SettingsCorridor):
    # no doc
    def Dispose(self):
        """ Dispose(self: DisposableWrapper, A_0: bool) """
        pass

    GridSettings = property(lambda self: object(), lambda self, v: None, lambda self: None)  # default
    """Get: GridSettings(self: SettingsCmdViewEditCorridorSection) -> SettingsCmdGridSettings

"""

    GridTextSettings = property(lambda self: object(), lambda self, v: None, lambda self: None)  # default
    """Get: GridTextSettings(self: SettingsCmdViewEditCorridorSection) -> SettingsCmdGridTextSettings

"""

    SectionSliderInMultipleViewports = property(lambda self: object(), lambda self, v: None, lambda self: None)  # default
    """Get: SectionSliderInMultipleViewports(self: SettingsCmdViewEditCorridorSection) -> SettingsCmdSectionSliderInMultipleViewports

"""

    ViewEditOptions = property(lambda self: object(), lambda self, v: None, lambda self: None)  # default
    """Get: ViewEditOptions(self: SettingsCmdViewEditCorridorSection) -> SettingsCmdViewEditOptions

"""


    SettingsCmdGridSettings = None
    SettingsCmdGridTextSettings = None
    SettingsCmdSectionSliderInMultipleViewports = None
    SettingsCmdViewEditOptions = None


class SettingsCmdVolumesDashboard(SettingsSurface):
    # no doc
    def Dispose(self):
        """ Dispose(self: DisposableWrapper, A_0: bool) """
        pass

    BoundedVolumeCreation = property(lambda self: object(), lambda self, v: None, lambda self: None)  # default
    """Get: BoundedVolumeCreation(self: SettingsCmdVolumesDashboard) -> SettingsCmdBoundedVolumeCreation

"""

    BuildOptions = property(lambda self: object(), lambda self, v: None, lambda self: None)  # default
    """Get: BuildOptions(self: SettingsCmdVolumesDashboard) -> SettingsCmdBuildOptions

"""

    DynamicHighlightOptions = property(lambda self: object(), lambda self, v: None, lambda self: None)  # default
    """Get: DynamicHighlightOptions(self: SettingsCmdVolumesDashboard) -> SettingsCmdDynamicHighlightOptions

"""

    VolumeSurfaceCreation = property(lambda self: object(), lambda self, v: None, lambda self: None)  # default
    """Get: VolumeSurfaceCreation(self: SettingsCmdVolumesDashboard) -> SettingsCmdVolumeSurfaceCreation

"""


    SettingsCmdBoundedVolumeCreation = None
    SettingsCmdBuildOptions = None
    SettingsCmdDynamicHighlightOptions = None
    SettingsCmdVolumeSurfaceCreation = None


class SettingsCmdWeedFeatures(SettingsGrading):
    # no doc
    def Dispose(self):
        """ Dispose(self: DisposableWrapper, A_0: bool) """
        pass

    FeatureLineWeed = property(lambda self: object(), lambda self, v: None, lambda self: None)  # default
    """Get: FeatureLineWeed(self: SettingsCmdWeedFeatures) -> SettingsCmdFeatureLineWeed

"""


    SettingsCmdFeatureLineWeed = None


class SettingsCoordinateSystem(object):
    # no doc
    Category = property(lambda self: object(), lambda self, v: None, lambda self: None)  # default
    """Get: Category(self: SettingsCoordinateSystem) -> str

"""

    Code = property(lambda self: object(), lambda self, v: None, lambda self: None)  # default
    """Get: Code(self: SettingsCoordinateSystem) -> str

"""

    Datum = property(lambda self: object(), lambda self, v: None, lambda self: None)  # default
    """Get: Datum(self: SettingsCoordinateSystem) -> str

"""

    Description = property(lambda self: object(), lambda self, v: None, lambda self: None)  # default
    """Get: Description(self: SettingsCoordinateSystem) -> str

"""

    Projection = property(lambda self: object(), lambda self, v: None, lambda self: None)  # default
    """Get: Projection(self: SettingsCoordinateSystem) -> str

"""

    Unit = property(lambda self: object(), lambda self, v: None, lambda self: None)  # default
    """Get: Unit(self: SettingsCoordinateSystem) -> str

"""



class SettingsDrawing(TreeOidWrapper):
    # no doc
    def Dispose(self):
        """ Dispose(self: DisposableWrapper, A_0: bool) """
        pass

    AbbreviationsSettings = property(lambda self: object(), lambda self, v: None, lambda self: None)  # default
    """Get: AbbreviationsSettings(self: SettingsDrawing) -> SettingsAbbreviation

"""

    AmbientSettings = property(lambda self: object(), lambda self, v: None, lambda self: None)  # default
    """Get: AmbientSettings(self: SettingsDrawing) -> SettingsAmbient

"""

    ApplyTransformSettings = property(lambda self: object(), lambda self, v: None, lambda self: None)  # default
    """Get: ApplyTransformSettings(self: SettingsDrawing) -> bool

Set: ApplyTransformSettings(self: SettingsDrawing) = value
"""

    ObjectLayerSettings = property(lambda self: object(), lambda self, v: None, lambda self: None)  # default
    """Get: ObjectLayerSettings(self: SettingsDrawing) -> SettingsObjectLayers

"""

    TransformationSettings = property(lambda self: object(), lambda self, v: None, lambda self: None)  # default
    """Get: TransformationSettings(self: SettingsDrawing) -> SettingsTransformation

"""

    UnitZoneSettings = property(lambda self: object(), lambda self, v: None, lambda self: None)  # default
    """Get: UnitZoneSettings(self: SettingsDrawing) -> SettingsUnitZone

"""



class SettingsLandXML(TreeOidWrapper):
    # no doc
    def Dispose(self):
        """ Dispose(self: DisposableWrapper, A_0: bool) """
        pass

    Export = property(lambda self: object(), lambda self, v: None, lambda self: None)  # default
    """Get: Export(self: SettingsLandXML) -> SettingsLandXMLExport

"""

    Import = property(lambda self: object(), lambda self, v: None, lambda self: None)  # default
    """Get: Import(self: SettingsLandXML) -> SettingsLandXMLImport

"""



class SettingsLandXMLExport(TreeOidWrapper):
    # no doc
    def Dispose(self):
        """ Dispose(self: DisposableWrapper, A_0: bool) """
        pass

    AlignmentExport = property(lambda self: object(), lambda self, v: None, lambda self: None)  # default
    """Get: AlignmentExport(self: SettingsLandXMLExport) -> SettingsAlignmentExport

"""

    Data = property(lambda self: object(), lambda self, v: None, lambda self: None)  # default
    """Get: Data(self: SettingsLandXMLExport) -> SettingsData

"""

    FeatureLineExport = property(lambda self: object(), lambda self, v: None, lambda self: None)  # default
    """Get: FeatureLineExport(self: SettingsLandXMLExport) -> SettingsFeatureLineExport

"""

    Identification = property(lambda self: object(), lambda self, v: None, lambda self: None)  # default
    """Get: Identification(self: SettingsLandXMLExport) -> SettingsIdentification

"""

    ParcelExport = property(lambda self: object(), lambda self, v: None, lambda self: None)  # default
    """Get: ParcelExport(self: SettingsLandXMLExport) -> SettingsParcelExport

"""

    PointExport = property(lambda self: object(), lambda self, v: None, lambda self: None)  # default
    """Get: PointExport(self: SettingsLandXMLExport) -> SettingsPointExport

"""

    SurfaceExport = property(lambda self: object(), lambda self, v: None, lambda self: None)  # default
    """Get: SurfaceExport(self: SettingsLandXMLExport) -> SettingsSurfaceExport

"""


    SettingsAlignmentExport = None
    SettingsData = None
    SettingsFeatureLineExport = None
    SettingsIdentification = None
    SettingsParcelExport = None
    SettingsPointExport = None
    SettingsSurfaceExport = None


class SettingsLandXMLImport(TreeOidWrapper):
    # no doc
    def Dispose(self):
        """ Dispose(self: DisposableWrapper, A_0: bool) """
        pass

    AlignmentImport = property(lambda self: object(), lambda self, v: None, lambda self: None)  # default
    """Get: AlignmentImport(self: SettingsLandXMLImport) -> SettingsAlignmentImport

"""

    ConflictResolution = property(lambda self: object(), lambda self, v: None, lambda self: None)  # default
    """Get: ConflictResolution(self: SettingsLandXMLImport) -> SettingsConflictResolution

"""

    DiameterUnits = property(lambda self: object(), lambda self, v: None, lambda self: None)  # default
    """Get: DiameterUnits(self: SettingsLandXMLImport) -> SettingsDiameterUnits

"""

    FeatureLineImport = property(lambda self: object(), lambda self, v: None, lambda self: None)  # default
    """Get: FeatureLineImport(self: SettingsLandXMLImport) -> SettingsFeatureLineImport

"""

    PipeNetwork = property(lambda self: object(), lambda self, v: None, lambda self: None)  # default
    """Get: PipeNetwork(self: SettingsLandXMLImport) -> SettingsPipeNetwork

"""

    PointImport = property(lambda self: object(), lambda self, v: None, lambda self: None)  # default
    """Get: PointImport(self: SettingsLandXMLImport) -> SettingsPointImport

"""

    PropertySetData = property(lambda self: object(), lambda self, v: None, lambda self: None)  # default
    """Get: PropertySetData(self: SettingsLandXMLImport) -> SettingsPropertySetData

"""

    Rotation = property(lambda self: object(), lambda self, v: None, lambda self: None)  # default
    """Get: Rotation(self: SettingsLandXMLImport) -> SettingsRotation

"""

    SurfaceImport = property(lambda self: object(), lambda self, v: None, lambda self: None)  # default
    """Get: SurfaceImport(self: SettingsLandXMLImport) -> SettingsSurfaceImport

"""

    Translation = property(lambda self: object(), lambda self, v: None, lambda self: None)  # default
    """Get: Translation(self: SettingsLandXMLImport) -> SettingsTranslation

"""


    SettingsAlignmentImport = None
    SettingsConflictResolution = None
    SettingsDiameterUnits = None
    SettingsFeatureLineImport = None
    SettingsPipeNetwork = None
    SettingsPointImport = None
    SettingsPropertySetData = None
    SettingsRotation = None
    SettingsSurfaceImport = None
    SettingsTranslation = None


class SettingsMassHaulLine(SettingsAmbient):
    # no doc
    def Dispose(self):
        """ Dispose(self: DisposableWrapper, A_0: bool) """
        pass


class SettingsMatchLine(SettingsAmbient):
    # no doc
    def Dispose(self):
        """ Dispose(self: DisposableWrapper, A_0: bool) """
        pass


class SettingsObjectLayer(TreeOidWrapper):
    # no doc
    def Dispose(self):
        """ Dispose(self: DisposableWrapper, A_0: bool) """
        pass

    LayerId = property(lambda self: object(), lambda self, v: None, lambda self: None)  # default
    """Get: LayerId(self: SettingsObjectLayer) -> ObjectId

Set: LayerId(self: SettingsObjectLayer) = value
"""

    LayerName = property(lambda self: object(), lambda self, v: None, lambda self: None)  # default
    """Get: LayerName(self: SettingsObjectLayer) -> str

Set: LayerName(self: SettingsObjectLayer) = value
"""

    Locked = property(lambda self: object(), lambda self, v: None, lambda self: None)  # default
    """Get: Locked(self: SettingsObjectLayer) -> bool

Set: Locked(self: SettingsObjectLayer) = value
"""

    Modifier = property(lambda self: object(), lambda self, v: None, lambda self: None)  # default
    """Get: Modifier(self: SettingsObjectLayer) -> ObjectLayerModifierType

Set: Modifier(self: SettingsObjectLayer) = value
"""

    ModifierValue = property(lambda self: object(), lambda self, v: None, lambda self: None)  # default
    """Get: ModifierValue(self: SettingsObjectLayer) -> str

Set: ModifierValue(self: SettingsObjectLayer) = value
"""

    ObjectType = property(lambda self: object(), lambda self, v: None, lambda self: None)  # default
    """Get: ObjectType(self: SettingsObjectLayer) -> SettingsObjectLayerType

"""



class SettingsObjectLayers(TreeOidWrapper):
    # no doc
    def Dispose(self):
        """ Dispose(self: DisposableWrapper, A_0: bool) """
        pass

    def GetObjectLayerSetting(self, settingsType):
        """ GetObjectLayerSetting(self: SettingsObjectLayers, settingsType: SettingsObjectLayerType) -> SettingsObjectLayer """
        pass

    ObjectControlledByLayer = property(lambda self: object(), lambda self, v: None, lambda self: None)  # default
    """Get: ObjectControlledByLayer(self: SettingsObjectLayers) -> bool

Set: ObjectControlledByLayer(self: SettingsObjectLayers) = value
"""



class SettingsObjectLayerType(Enum):
    """ enum SettingsObjectLayerType, values: Alignment (0), AlignmentLabeling (1), AlignmentTable (2), Appurtenance (56), AppurtenanceLabeling (57), Assembly (3), BuildingSite (53), CantView (58), Catchment (59), CatchmentLabeling (60), Corridor (4), CorridorSection (5), FeatureLine (6), Fitting (61), FittingLabeling (62), GeneralNoteLabel (7), GeneralSegmentLabel (8), Grading (9), GradingLabeling (10), GridSurface (11), GridSurfaceLabeling (12), Interference (13), Intersection (54), IntersectionLabeling (55), MassHaulLine (14), MassHaulView (15), MatchLine (16), MatchLineLabeling (17), MaterialSection (18), MaterialTable (19), Parcel (20), ParcelLabeling (21), ParcelSegment (22), ParcelSegmentLabeling (23), ParcelTable (24), Pipe (25), PipeAndStructureTable (27), PipeLabeling (26), PipeNetworkSection (28), PipeOrStructureProfile (29), PointTable (30), PressureNetworkSection (63), PressurePartProfile (64), PressurePartTable (65), PressurePipe (66), PressurePipeLabeling (67), Profile (31), ProfileLabeling (32), ProfileView (33), ProfileViewLabeling (34), SampleLine (35), SampleLineLabeling (36), Section (37), SectionLabeling (38), SectionView (39), SectionViewLabeling (40), SectionViewQuantityTakeoffTable (41), Sheet (42), Structure (43), StructureLabeling (44), Subassembly (45), SuperelevationView (68), SurfaceLegendTable (46), SurveyFigure (47), SurveyFigureLabeling (69), SurveyFigureSegmentLable (70), SurveyNetwork (48), TinSurface (49), TinSurfaceLabeling (50), ViewFrame (51), ViewFrameLabeling (52) """
    Alignment = None
    AlignmentLabeling = None
    AlignmentTable = None
    Appurtenance = None
    AppurtenanceLabeling = None
    Assembly = None
    BuildingSite = None
    CantView = None
    Catchment = None
    CatchmentLabeling = None
    Corridor = None
    CorridorSection = None
    FeatureLine = None
    Fitting = None
    FittingLabeling = None
    GeneralNoteLabel = None
    GeneralSegmentLabel = None
    Grading = None
    GradingLabeling = None
    GridSurface = None
    GridSurfaceLabeling = None
    Interference = None
    Intersection = None
    IntersectionLabeling = None
    MassHaulLine = None
    MassHaulView = None
    MatchLine = None
    MatchLineLabeling = None
    MaterialSection = None
    MaterialTable = None
    Parcel = None
    ParcelLabeling = None
    ParcelSegment = None
    ParcelSegmentLabeling = None
    ParcelTable = None
    Pipe = None
    PipeAndStructureTable = None
    PipeLabeling = None
    PipeNetworkSection = None
    PipeOrStructureProfile = None
    PointTable = None
    PressureNetworkSection = None
    PressurePartProfile = None
    PressurePartTable = None
    PressurePipe = None
    PressurePipeLabeling = None
    Profile = None
    ProfileLabeling = None
    ProfileView = None
    ProfileViewLabeling = None
    SampleLine = None
    SampleLineLabeling = None
    Section = None
    SectionLabeling = None
    SectionView = None
    SectionViewLabeling = None
    SectionViewQuantityTakeoffTable = None
    Sheet = None
    Structure = None
    StructureLabeling = None
    Subassembly = None
    SuperelevationView = None
    SurfaceLegendTable = None
    SurveyFigure = None
    SurveyFigureLabeling = None
    SurveyFigureSegmentLable = None
    SurveyNetwork = None
    TinSurface = None
    TinSurfaceLabeling = None
    value__ = None
    ViewFrame = None
    ViewFrameLabeling = None


class SettingsPipe(SettingsAmbient):
    # no doc
    def Dispose(self):
        """ Dispose(self: DisposableWrapper, A_0: bool) """
        pass


class SettingsPressureAppurtenance(SettingsAmbient):
    # no doc
    def Dispose(self):
        """ Dispose(self: DisposableWrapper, A_0: bool) """
        pass


class SettingsPressureFitting(SettingsAmbient):
    # no doc
    def Dispose(self):
        """ Dispose(self: DisposableWrapper, A_0: bool) """
        pass


class SettingsPressurePipe(SettingsAmbient):
    # no doc
    def Dispose(self):
        """ Dispose(self: DisposableWrapper, A_0: bool) """
        pass


class SettingsRoot(TreeOidWrapper):
    # no doc
    def Dispose(self):
        """ Dispose(self: DisposableWrapper, A_0: bool) """
        pass

    def GetSettings(self):
# Error generating skeleton for function GetSettings: Method must be called on a Type for which Type.IsGenericParameter is false.

    AssociateShortcutProjectId = property(lambda self: object(), lambda self, v: None, lambda self: None)  # default
    """Get: AssociateShortcutProjectId(self: SettingsRoot) -> str

Set: AssociateShortcutProjectId(self: SettingsRoot) = value
"""

    DrawingSettings = property(lambda self: object(), lambda self, v: None, lambda self: None)  # default
    """Get: DrawingSettings(self: SettingsRoot) -> SettingsDrawing

"""

    LandXMLSettings = property(lambda self: object(), lambda self, v: None, lambda self: None)  # default
    """Get: LandXMLSettings(self: SettingsRoot) -> SettingsLandXML

"""

    TagSettings = property(lambda self: object(), lambda self, v: None, lambda self: None)  # default
    """Get: TagSettings(self: SettingsRoot) -> SettingsTag

"""



class SettingsSection(SettingsAmbient):
    # no doc
    def Dispose(self):
        """ Dispose(self: DisposableWrapper, A_0: bool) """
        pass

    NameFormat = property(lambda self: object(), lambda self, v: None, lambda self: None)  # default
    """Get: NameFormat(self: SettingsSection) -> SettingsNameFormat

"""

    Styles = property(lambda self: object(), lambda self, v: None, lambda self: None)  # default
    """Get: Styles(self: SettingsSection) -> SettingsStyles

"""


    SettingsNameFormat = None
    SettingsStyles = None


class SettingsStructure(SettingsAmbient):
    # no doc
    def Dispose(self):
        """ Dispose(self: DisposableWrapper, A_0: bool) """
        pass


class SettingsTag(TreeOidWrapper):
    # no doc
    def Dispose(self):
        """ Dispose(self: DisposableWrapper, A_0: bool) """
        pass

    Creation = property(lambda self: object(), lambda self, v: None, lambda self: None)  # default
    """Get: Creation(self: SettingsTag) -> SettingsCreation

"""

    Renumbering = property(lambda self: object(), lambda self, v: None, lambda self: None)  # default
    """Get: Renumbering(self: SettingsTag) -> SettingsRenumbering

"""


    SettingsCreation = None
    SettingsRenumbering = None


class SettingsTransformation(TreeOidWrapper):
    # no doc
    def Dispose(self):
        """ Dispose(self: DisposableWrapper, A_0: bool) """
        pass

    ApplySeaLevelScaleFactor = property(lambda self: object(), lambda self, v: None, lambda self: None)  # default
    """Get: ApplySeaLevelScaleFactor(self: SettingsTransformation) -> bool

Set: ApplySeaLevelScaleFactor(self: SettingsTransformation) = value
"""

    GridReferencePoint = property(lambda self: object(), lambda self, v: None, lambda self: None)  # default
    """Get: GridReferencePoint(self: SettingsTransformation) -> Point2d

Set: GridReferencePoint(self: SettingsTransformation) = value
"""

    GridRotationPoint = property(lambda self: object(), lambda self, v: None, lambda self: None)  # default
    """Get: GridRotationPoint(self: SettingsTransformation) -> Point2d

Set: GridRotationPoint(self: SettingsTransformation) = value
"""

    GridScaleFactor = property(lambda self: object(), lambda self, v: None, lambda self: None)  # default
    """Get: GridScaleFactor(self: SettingsTransformation) -> float

Set: GridScaleFactor(self: SettingsTransformation) = value
"""

    GridScaleFactorComputation = property(lambda self: object(), lambda self, v: None, lambda self: None)  # default
    """Get: GridScaleFactorComputation(self: SettingsTransformation) -> GridScaleFactorType

Set: GridScaleFactorComputation(self: SettingsTransformation) = value
"""

    LocalReferencePoint = property(lambda self: object(), lambda self, v: None, lambda self: None)  # default
    """Get: LocalReferencePoint(self: SettingsTransformation) -> Point2d

Set: LocalReferencePoint(self: SettingsTransformation) = value
"""

    LocalRotationPoint = property(lambda self: object(), lambda self, v: None, lambda self: None)  # default
    """Get: LocalRotationPoint(self: SettingsTransformation) -> Point2d

Set: LocalRotationPoint(self: SettingsTransformation) = value
"""

    RotationToGridAzimuth = property(lambda self: object(), lambda self, v: None, lambda self: None)  # default
    """Get: RotationToGridAzimuth(self: SettingsTransformation) -> float

Set: RotationToGridAzimuth(self: SettingsTransformation) = value
"""

    RotationToGridNorth = property(lambda self: object(), lambda self, v: None, lambda self: None)  # default
    """Get: RotationToGridNorth(self: SettingsTransformation) -> float

Set: RotationToGridNorth(self: SettingsTransformation) = value
"""

    SeaLevelScaleElevation = property(lambda self: object(), lambda self, v: None, lambda self: None)  # default
    """Get: SeaLevelScaleElevation(self: SettingsTransformation) -> float

Set: SeaLevelScaleElevation(self: SettingsTransformation) = value
"""

    SpecifyRotationType = property(lambda self: object(), lambda self, v: None, lambda self: None)  # default
    """Get: SpecifyRotationType(self: SettingsTransformation) -> SpecifyRotationType

Set: SpecifyRotationType(self: SettingsTransformation) = value
"""

    SpheroidRadius = property(lambda self: object(), lambda self, v: None, lambda self: None)  # default
    """Get: SpheroidRadius(self: SettingsTransformation) -> float

"""



class SettingsUnitZone(TreeOidWrapper):
    # no doc
    def Dispose(self):
        """ Dispose(self: DisposableWrapper, A_0: bool) """
        pass

    @staticmethod
    def GetAllCodes():
        """ GetAllCodes() -> Array[str] """
        pass

    @staticmethod
    def GetCoordinateSystemByCode(code):
        """ GetCoordinateSystemByCode(code: str) -> SettingsCoordinateSystem """
        pass

    AngularUnits = property(lambda self: object(), lambda self, v: None, lambda self: None)  # default
    """Get: AngularUnits(self: SettingsUnitZone) -> AngleUnitType

Set: AngularUnits(self: SettingsUnitZone) = value
"""

    CoordinateSystemCode = property(lambda self: object(), lambda self, v: None, lambda self: None)  # default
    """Get: CoordinateSystemCode(self: SettingsUnitZone) -> str

Set: CoordinateSystemCode(self: SettingsUnitZone) = value
"""

    DrawingScale = property(lambda self: object(), lambda self, v: None, lambda self: None)  # default
    """Get: DrawingScale(self: SettingsUnitZone) -> float

Set: DrawingScale(self: SettingsUnitZone) = value
"""

    DrawingUnits = property(lambda self: object(), lambda self, v: None, lambda self: None)  # default
    """Get: DrawingUnits(self: SettingsUnitZone) -> DrawingUnitType

Set: DrawingUnits(self: SettingsUnitZone) = value
"""

    ImperialToMetricConversion = property(lambda self: object(), lambda self, v: None, lambda self: None)  # default
    """Get: ImperialToMetricConversion(self: SettingsUnitZone) -> ImperialToMetricConversionType

Set: ImperialToMetricConversion(self: SettingsUnitZone) = value
"""

    MatchAutoCADVariables = property(lambda self: object(), lambda self, v: None, lambda self: None)  # default
    """Get: MatchAutoCADVariables(self: SettingsUnitZone) -> bool

Set: MatchAutoCADVariables(self: SettingsUnitZone) = value
"""

    ScaleObjectsFromOtherDrawings = property(lambda self: object(), lambda self, v: None, lambda self: None)  # default
    """Get: ScaleObjectsFromOtherDrawings(self: SettingsUnitZone) -> bool

Set: ScaleObjectsFromOtherDrawings(self: SettingsUnitZone) = value
"""



class SettingsViewFrame(SettingsAmbient):
    # no doc
    def Dispose(self):
        """ Dispose(self: DisposableWrapper, A_0: bool) """
        pass


class SpecifyRotationType(Enum):
    """ enum SpecifyRotationType, values: GridRotationAngle (1), RotationPoint (0) """
    GridRotationAngle = None
    RotationPoint = None
    value__ = None


class TableAnchorType(Enum):
    """ enum TableAnchorType, values: BottomCenter (7), BottomLeft (6), BottomRight (8), MiddleCenter (4), MiddleLeft (3), MiddleRight (5), TopCenter (1), TopLeft (0), TopRight (2) """
    BottomCenter = None
    BottomLeft = None
    BottomRight = None
    MiddleCenter = None
    MiddleLeft = None
    MiddleRight = None
    TopCenter = None
    TopLeft = None
    TopRight = None
    value__ = None


class TableLayoutType(Enum):
    """ enum TableLayoutType, values: Horizontal (0), Vertical (1) """
    Horizontal = None
    value__ = None
    Vertical = None


class TileDirectionType(Enum):
    """ enum TileDirectionType, values: Across (0), Down (1) """
    Across = None
    Down = None
    value__ = None


