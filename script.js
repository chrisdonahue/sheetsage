window.ismir = window.ismir || {};

(function (Tone, ns) {
  const ROOT = "https://chrisdonahue.com/ismir2022/static";

  const RWC_RYY_UIDS = [
    "G002",
    "G010",
    "G036",
    "G068",
    "G072",
    "P012",
    "P038",
    "P060",
    "P070",
    "P079",
  ];

  const RWC_RYY_METHODS = [
    ["ground_truth", "Ground Truth"],
    ["mt3", "MT3"],
    ["melodia", "Melodia"],
    ["spleeter_tony", "Spleeter + Tony"],
    ["ryynanen", "Ryynanen"],
    ["our_mel", "Our Mel"],
    ["our_mt3", "Our MT3"],
    ["our_jkb", "Our Jukebox"],
  ];

  let PLAYER;
  let ACTIVE_EXAMPLE;

  function _urlToUid(url) {
    url = url || "";
    const urlSplit = url.split("/");
    let result = null;
    if (urlSplit.length > 0) {
      result = urlSplit[urlSplit.length - 1];
    }
    return result;
  }

  async function setActiveExample(a, b) {
    let offset = null;
    if (_urlToUid(ACTIVE_EXAMPLE) === _urlToUid(a)) {
      offset = PLAYER.getTransportOffset();
    }
    const loaded = await PLAYER.setSources(a, b);
    if (loaded) {
      if (offset !== null) PLAYER.setTransportOffset(offset);
      PLAYER.play();
      ACTIVE_EXAMPLE = a;
    } else {
      alert(
        "Failed to load sound example... please let the authors know and try again later!"
      );
    }
  }

  async function onDomReady() {
    PLAYER = new ns.XfadePlayer(
      document.getElementById("toggle"),
      document.getElementById("stop"),
      document.getElementById("transport"),
      document.getElementById("xfade"),
      document.getElementById("volume")
    );

    const rwcRyyTableBodyEl = document.getElementById("rwc-ryy-table-body");
    const rwcRyyMethodTemplate =
      document.getElementById("rwc-ryy-method").content;
    const rwcRyyExampleTemplate =
      document.getElementById("rwc-ryy-example").content;
    for (let i = 0; i < RWC_RYY_METHODS.length; ++i) {
      const [method, methodHtml] = RWC_RYY_METHODS[i];
      const methodEl = rwcRyyMethodTemplate.cloneNode(true).querySelector("tr");
      const methodNameEl = document.createElement("td");
      methodNameEl.innerHTML = methodHtml;
      methodNameEl.className = "method-name";
      methodEl.appendChild(methodNameEl);
      for (let j = 0; j < RWC_RYY_UIDS.length; ++j) {
        const uid = RWC_RYY_UIDS[j];
        const exampleEl = rwcRyyExampleTemplate.cloneNode(true);
        const inputEl = exampleEl.querySelector("input");
        inputEl.onchange = () => {
          if (inputEl.checked) {
            setActiveExample(
              `${ROOT}/rwc_ryy/input/${uid}.mp3`,
              `${ROOT}/rwc_ryy/output/${method}/${uid}.mp3`
            );
          }
        };
        methodEl.appendChild(exampleEl);
      }
      rwcRyyTableBodyEl.appendChild(methodEl);
    }
  }

  async function init() {}

  document.addEventListener("DOMContentLoaded", onDomReady, false);
  init();
})(window.Tone, window.ismir);
