/*
 * Small notifications used across the DroneAI pages.
 *
 *   showToast({ type: 'error', message: 'Could not read that file' })
 *   showToast({ type: 'error', title: "Couldn't load the video",
 *               message: reason, detail: rawError, duration: 0 })
 *
 * type: success | error | warning | info   (sets the colour of the left edge)
 * duration: ms, or 0 to leave it up until dismissed.
 *
 * Most toasts only need `message`. Add a `title` when the message is a
 * separate piece of information, not a restatement of the title.
 */
(function () {
  function stack() {
    let el = document.getElementById("toast-stack");
    if (!el) {
      el = document.createElement("div");
      el.id = "toast-stack";
      document.body.appendChild(el);
    }
    return el;
  }

  function dismiss(toast) {
    if (!toast || toast.dataset.closing === "1") return;
    toast.dataset.closing = "1";
    toast.classList.remove("toast--in");
    setTimeout(() => toast.remove(), 240);
  }

  window.showToast = function (opts) {
    opts = opts || {};
    const type = opts.type || "info";
    const duration = opts.duration === undefined
      ? (type === "error" ? 12000 : 5000)
      : opts.duration;

    const toast = document.createElement("div");
    toast.className = "toast toast--" + type;

    const body = document.createElement("div");
    body.className = "toast__body";

    if (opts.title) {
      const title = document.createElement("div");
      title.className = "toast__title";
      title.textContent = opts.title;
      body.appendChild(title);
    }

    if (opts.message) {
      const msg = document.createElement("div");
      // Without a title the message carries the toast, so give it full weight.
      msg.className = opts.title ? "toast__msg" : "toast__title";
      msg.textContent = opts.message;
      body.appendChild(msg);
    }

    if (opts.detail) {
      const detail = document.createElement("div");
      detail.className = "toast__detail";
      detail.textContent = opts.detail;
      body.appendChild(detail);
    }

    const close = document.createElement("button");
    close.className = "toast__close";
    close.type = "button";
    close.setAttribute("aria-label", "Dismiss");
    close.textContent = "×";
    close.addEventListener("click", () => dismiss(toast));

    toast.appendChild(body);
    toast.appendChild(close);
    stack().appendChild(toast);

    // Let the browser paint the initial state before animating in.
    requestAnimationFrame(() => toast.classList.add("toast--in"));

    if (duration > 0) {
      setTimeout(() => dismiss(toast), duration);
    }

    return toast;
  };
})();
