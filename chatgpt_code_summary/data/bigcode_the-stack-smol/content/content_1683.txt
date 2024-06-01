from decimal import Decimal
from django.db import models
from polymorphic.models import PolymorphicModel
from django.utils.translation import ugettext_lazy as _
from django.contrib.auth.models import User
from django.contrib.contenttypes.models import ContentType
from django.contrib.contenttypes.fields import GenericForeignKey
from auction.utils.loader import get_model_string
from django.conf import settings


class CurrencyField(models.DecimalField):

    def to_python(self, value):
        try:
            return super(CurrencyField, self).to_python(value=value).quantize(Decimal("0.01"))
        except AttributeError:
            return None


class BaseAuction(PolymorphicModel):
    name = models.CharField(max_length=255, verbose_name=_('Auction name'))
    slug = models.SlugField(unique=True, verbose_name=_('Slug'))
    start_date = models.DateTimeField(verbose_name=_('Start date'))
    end_date = models.DateTimeField(verbose_name=_('End date'))
    active = models.BooleanField(default=False, verbose_name=_('Active'))
    total_bids = models.IntegerField(default=0, verbose_name=_('Total bids'))
    date_added = models.DateTimeField(auto_now_add=True, verbose_name=_('Date added'))
    last_modified = models.DateTimeField(auto_now=True, verbose_name=_('Last modified'))

    class Meta:
        abstract = True
        app_label = 'auction'
        verbose_name = _('Auction')
        verbose_name_plural = _('Auctions')

    def __unicode__(self):
        return self.name


class BaseBidBasket(models.Model):
    """
    This models functions similarly to a shopping cart, except it expects a logged in user.
    """
    user = models.OneToOneField(User, on_delete=models.CASCADE, related_name="%(app_label)s_%(class)s_related", verbose_name=_('User'))
    date_added = models.DateTimeField(auto_now_add=True, verbose_name=_('Date added'))
    last_modified = models.DateTimeField(auto_now=True, verbose_name=_('Last modified'))

    class Meta:
        abstract = True
        app_label = 'auction'
        verbose_name = _('Bid basket')
        verbose_name_plural = _('Bid baskets')

    def add_bid(self, lot, amount):
        from auction.models import BidItem
        self.save()

        if not lot.is_biddable:
            return False

        try:
            amount = Decimal(amount)
        except Exception as e:
            amount = Decimal('0')

        from auction.models.lot import Lot
        item,created = BidItem.objects.get_or_create(bid_basket=self,
                                                     content_type=ContentType.objects.get_for_model(Lot),
                                                     lot_id=lot.pk)
        if item:
            item.amount=amount
            item.save()
        return item

    def update_bid(self, bid_basket_item_id, amount):
        """
        Update amount of bid. Delete bid if amount is 0.
        """

        try:
            amount = Decimal(amount)
        except Exception as e:
            amount = Decimal('0')

        bid_basket_item = self.bids.get(pk=bid_basket_item_id)
        if not bid_basket_item.is_locked():
            if amount == 0:
                bid_basket_item.delete()
            else:
                bid_basket_item.amount = amount
                bid_basket_item.save()
            self.save()
        return bid_basket_item

    def delete_bid(self, bid_basket_item_id):
        """
        Delete a single item from bid basket.
        """
        bid_basket_item = self.bids.get(pk=bid_basket_item_id)
        if not bid_basket_item.is_locked():
            bid_basket_item.delete()
        return bid_basket_item

    def empty(self):
        """
        Remove all bids from bid basket.
        """
        if self.pk:
            bids = self.bids.all()
            for bid in bids:
                if not bid.is_locked():
                    bid.delete()

    @property
    def bids(self):
        """
        Used as accessor for abstract related (BaseBidItem.bid_items).

        If you override BaseBidItem and use a label other than "auction"
        you will also need to set AUCTION_BIDBASKET_BIDS_RELATED_NAME.
        Example: foo_biditem_related
                 (where your label is "foo" and your model is "BidItem")
        """
        bids = getattr(settings, 'AUCTION_BIDBASKET_BIDS_RELATED_NAME',
                       'auction_biditem_related')
        return getattr(self, bids)

    @property
    def total_bids(self):
        """
        Returns total bids in basket.
        """
        return len(self.bids.all())

class BaseAuctionLot(PolymorphicModel):
    name = models.CharField(max_length=255, verbose_name=_('Lot name'))
    slug = models.SlugField(auto_created=True, verbose_name=_('Slug'))
    active = models.BooleanField(default=False, verbose_name=_('Active'))
    is_biddable = models.BooleanField(default=False, verbose_name=_('Is biddable?'))
    content_type = models.ForeignKey(ContentType, on_delete=models.CASCADE, related_name="%(app_label)s_%(class)s_lots",
                                     verbose_name=_('Content type'))
    object_id = models.PositiveIntegerField(verbose_name=_('Object ID'))
    content_object = GenericForeignKey('content_type', 'object_id')
    date_added = models.DateTimeField(auto_now_add=True, verbose_name=_('Date added'))
    last_modified = models.DateTimeField(auto_now=True, verbose_name=_('Last modified'))

    class Meta:
        abstract = True
        app_label = 'auction'
        verbose_name = _('Auction lot')
        verbose_name_plural = _('Auction lots')

    def __unicode__(self):
        return self.name

    @property
    def is_locked(self):
        """
        This property is meant to be overwritten with your own logic. Bid baskets
        check this method to find out if a bid can be manipulated.
        """
        import auction.utils.generic
        now = auction.utils.generic.get_current_time()
        return self.content_object.end_date <= now

class BaseBidItem(models.Model):
    """
    This is a holder for total number of bids and a pointer to
    item being bid on.
    """

    bid_basket = models.ForeignKey(get_model_string("BidBasket"), on_delete=models.CASCADE, related_name="%(app_label)s_%(class)s_related", verbose_name=_('Bid basket'))
    content_type = models.ForeignKey(ContentType, on_delete=models.CASCADE, related_name="%(app_label)s_%(class)s_related", verbose_name=_('Content type'))
    lot_id = models.PositiveIntegerField(verbose_name=_('Lot ID'))
    lot_object = GenericForeignKey('content_type', 'lot_id')
    amount = CurrencyField(max_digits=10, decimal_places=2, null=True, blank=True, verbose_name=_('Amount'))

    class Meta:
        abstract = True
        app_label = 'auction'
        verbose_name = _('Bid item')
        verbose_name_plural = _('Bid items')

    def is_locked(self):
        return self.lot.is_locked

    @property
    def lot(self):
        return self.lot_object