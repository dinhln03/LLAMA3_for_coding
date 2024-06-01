from whey.mixin import BuilderMixin

class GettextMixin:
	def build_messages(self: BuilderMixin):
		from babel.messages.mofile import write_mo
		from babel.messages.pofile import read_po

		locales = self.pkgdir / "locales"
		if self.verbose:
			print("  Building messages")

		for po in locales.glob("*/LC_MESSAGES/pm2hw.po"):
			with po.open("rt", encoding="UTF-8") as f:
				catalog = read_po(f, po.parts[-3], po.stem)

			mo = self.build_dir / po.relative_to(self.project_dir).with_suffix(".mo")
			mo.parent.maybe_make(parents=True)
			with mo.open("wb") as f:
				write_mo(f, catalog)
			self.report_written(mo)

			if self.verbose:
				print("    Wrote language file:", mo)


# class SDistBuilder(GettextMixin, builder.SDistBuilder):
# 	def call_additional_hooks(self):
# 		self.build_messages()


# class WheelBuilder(GettextMixin, builder.WheelBuilder):
# 	def call_additional_hooks(self):
# 		self.build_messages()
