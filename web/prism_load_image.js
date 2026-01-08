import { app } from "../../scripts/app.js";

// Extension pour ajouter un bouton "Update Preview" aux nodes PrismLoadImage et PrismLoadAsset
app.registerExtension({
    name: "Comfy.PrismLoadImage.UpdatePreview",

    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData.name === "PrismLoadImage" || nodeData.name === "PrismLoadAsset") {

            // Ajouter l'option "Update Preview" dans le menu contextuel
            const getExtraMenuOptions = nodeType.prototype.getExtraMenuOptions;
            nodeType.prototype.getExtraMenuOptions = function(_, options) {
                if (getExtraMenuOptions) {
                    getExtraMenuOptions.apply(this, arguments);
                }

                options.unshift({
                    content: "🔄 Update Preview",
                    callback: () => {
                        // Trouver le widget force_refresh
                        const forceRefreshWidget = this.widgets?.find(w => w.name === "force_refresh");

                        if (forceRefreshWidget) {
                            // Mettre à jour avec un timestamp pour forcer le recalcul
                            forceRefreshWidget.value = Date.now().toString();

                            // Marquer le node comme nécessitant un recalcul
                            this.setDirtyCanvas(true, true);

                            console.log("Preview update triggered for PrismLoadImage");
                        }
                    }
                });
            };

            // Ajouter également un bouton visible dans le node
            const onNodeCreated = nodeType.prototype.onNodeCreated;
            nodeType.prototype.onNodeCreated = function () {
                const result = onNodeCreated ? onNodeCreated.apply(this, arguments) : undefined;

                // Ajouter un bouton visible
                const button = this.addWidget(
                    "button",
                    "update_preview",
                    "🔄 Update Preview",
                    () => {
                        const forceRefreshWidget = this.widgets?.find(w => w.name === "force_refresh");
                        if (forceRefreshWidget) {
                            forceRefreshWidget.value = Date.now().toString();
                            console.log("Preview update triggered via button");
                        }
                    },
                    {}
                );

                // S'assurer que le bouton n'est pas sérialisé dans le workflow
                if (button) {
                    button.serialize = false;
                }

                return result;
            };
        }
    }
});
