import { app } from "../../scripts/app.js";
import { ComfyWidgets } from "../../scripts/widgets.js";

app.registerExtension({
    name: "NanoBanana.TextDisplay",
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData.name === "NanoBananaTextDisplay") {

            const onNodeCreated = nodeType.prototype.onNodeCreated;
            nodeType.prototype.onNodeCreated = function () {
                if (onNodeCreated) {
                    onNodeCreated.apply(this, arguments);
                }

                const w = ComfyWidgets["STRING"](this, "text_output", ["STRING", { multiline: true }], app).widget;
                w.inputEl.readOnly = true;
                w.inputEl.style.opacity = 0.8;
                this.text_widget = w;

                this.setSize(this.computeSize());
            };

            const onExecuted = nodeType.prototype.onExecuted;
            nodeType.prototype.onExecuted = function (message) {
                if (onExecuted) {
                    onExecuted.apply(this, arguments);
                }

                const textValue = (message && message.text) ? message.text[0] || "" : "";

                if (this.text_widget) {
                    this.text_widget.value = textValue;
                    if (this.text_widget.inputEl) {
                        this.text_widget.inputEl.value = textValue;
                    }
                } else if (this.widgets) {
                    let found = false;
                    for (let i = 0; i < this.widgets.length; i++) {
                        if (this.widgets[i].name === "text_output") {
                            this.widgets[i].value = textValue;
                            if (this.widgets[i].inputEl) {
                                this.widgets[i].inputEl.value = textValue;
                            }
                            found = true;
                            break;
                        }
                    }
                    if (!found) {
                        const w = ComfyWidgets["STRING"](this, "text_output", ["STRING", { multiline: true }], app).widget;
                        w.inputEl.readOnly = true;
                        w.inputEl.style.opacity = 0.8;
                        w.value = textValue;
                        if (w.inputEl) {
                            w.inputEl.value = textValue;
                        }
                    }
                }

                // Trigger resize and re-render
                this.setSize(this.computeSize());
                if (app.canvas) {
                    app.canvas.setDirty(true, true);
                }
            };
        }
    }
});
