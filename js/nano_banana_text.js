import { app } from "../../scripts/app.js";
import { ComfyWidgets } from "../../scripts/widgets.js";

app.registerExtension({
    name: "NanoBanana.TextDisplay",
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData.name === "NanoBananaTextDisplay") {
            const onExecuted = nodeType.prototype.onExecuted;
            nodeType.prototype.onExecuted = function (message) {
                onExecuted?.apply(this, arguments);

                if (this.widgets) {
                    for (let i = 0; i < this.widgets.length; i++) {
                        if (this.widgets[i].name === "text_output") {
                            this.widgets[i].value = message.text[0] || "";
                            return;
                        }
                    }
                }

                // If we get here, the widget hasn't been created yet
                let textValue = "";
                if (message && message.text) {
                    textValue = message.text[0] || "";
                }
                const w = ComfyWidgets["STRING"](this, "text_output", ["STRING", { multiline: true }], app).widget;
                w.inputEl.readOnly = true;
                w.inputEl.style.opacity = 0.8;
                w.value = textValue;
            };
        }
    }
});
