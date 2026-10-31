ZOLA = nix develop ./runtime --command zola

.PHONY: main serve draft

main:
	$(ZOLA) build

serve:
	$(ZOLA) serve --open

draft:
	$(ZOLA) serve --open --drafts
