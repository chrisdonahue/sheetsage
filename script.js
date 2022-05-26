window.ismir = window.ismir || {};

(function (Tone, ns) {
  const ROOT = "https://chrisdonahue.com/ismir2022/static";

  const RWC_RYY_UIDS = [
    ["G002", ""],
    ["G010", ""],
    ["G036", ""],
    ["G068", ""],
    ["G072", ""],
    ["P012", ""],
    ["P038", ""],
    ["P060", ""],
    ["P070", ""],
    ["P079", ""],
  ];

  const RWC_RYY_METHODS_F1 = [
    ["ground_truth", "Ground Truth", "1.000"],
    [null, null, null],
    ["mt3", "MT3", "0.133"],
    ["melodia", "Melodia", "0.201"],
    ["spleeter_tony", "Spleeter + Tony", "0.341"],
    ["ryynanen", "Ryynanen", "<b>0.420</b>"],
    [null, null, null],
    ["our_mel", "Our Mel", "0.631"],
    ["our_mt3", "Our MT3", "0.701"],
    ["our_jkb", "Our Jukebox", "<b>0.744</b>"],
  ];

  const HOOKTHEORY_TEST_UIDS = [
    // display
    ["ANmprGXPxyM", "YD7NFY_EoFk"],
    ["JNgqKvWeorz", "EUawDUtu0H4"],
    ["ZbgOPEkwxnY", "OSumeNU3zn0"],
    ["KexEq_pvx_B", "em0MknB6wFo"],
    ["KexEqqYXx_B", "9Y6H-YjsE9Q"],
    ["LAgQBkXvxyO", "5ghA8QIUNOo"],
    ["jDgXVDpdoKl", "dK9eLe8EQps"],
    ["YAg-EZZVgle", "WZlYGN5W2Yg"],
    ["ZbgOOPrbgnY", "nJ0Nk07_5po"],
    ["d_gwrZVMmGV", "S7sN-cFhwuk"],
    // backups
    ["JNgqlr-Jmrz", "LK2Urb_naAM"],
    ["ZOxVbvRGxdq", "tFOZWRknYSI"],
    ["ZbgOPEkwxnY", "OSumeNU3zn0"],
    ["jDgXVDpdoKl", "dK9eLe8EQps"],
    ["VMgJkMqExqK", "KDKva-s_khY"],
    ["ROmNWpwAgNw", "BYY1iTyWa54"],
    ["Dpgv-pekmad", "3exsRhw3xt8"],
    ["AQodYb-WmDl", "pTOC_q0NLTk"],
    ["YAg-ELPBgle", "8tlZJvTijhY"],
    ["YAg-aNMBgle", "SXKGawGf4Ls"],
    ["YAg-kyjwole", "-DIiMdc5WIs"],
    ["d_gwjNYbmGV", "cNMXSKfWfLQ"],
  ];

  const HOOKTHEORY_TEST_METHODS_F1 = [
    ["ground_truth", "Ground Truth", "1.000"],
    [null, null, null],
    ["our_mel", "Mel", "0.514"],
    ["our_mt3", "MT3", "0.550"],
    ["our_jkb", "Jukebox", "<b>0.615</b>"],
    [null, null, null],
    ["our_melmt3", "Mel + MT3", "0.548"],
    ["our_meljkb", "Mel + Jukebox", "0.617"],
    ["our_jkbmt3", "MT3 + Jukebox", "0.622"],
    ["our_melmt3jkb", "Mel + MT3 + Jukebox", "<b>0.623</b>"],
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
    document.getElementById("footer").style.display = "table";
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

    const radioTableMethodTemplate =
      document.getElementById("radio-table-method").content;
    const radioTableHrTemplate =
      document.getElementById("radio-table-hr").content;
    const radioTableExampleTemplate = document.getElementById(
      "radio-table-example"
    ).content;

    const tables = [
      [
        document.getElementById("rwc-ryy-table-body"),
        RWC_RYY_METHODS_F1,
        RWC_RYY_UIDS,
        `${ROOT}/rwc_ryy`,
      ],
      [
        document.getElementById("hooktheory-test-table-body"),
        HOOKTHEORY_TEST_METHODS_F1,
        HOOKTHEORY_TEST_UIDS,
        `${ROOT}/hooktheory_test`,
      ],
    ];

    for (let t = 0; t < tables.length; ++t) {
      const [tableBodyEl, methods, uids, root] = tables[t];

      for (let m = 0; m < methods.length; ++m) {
        const [method, methodHtml, methodF1] = methods[m];

        let methodEl;
        if (method === null) {
          methodEl = radioTableHrTemplate.cloneNode(true).querySelector("tr");
        } else {
          methodEl = radioTableMethodTemplate
            .cloneNode(true)
            .querySelector("tr");
          const methodNameEl = document.createElement("td");
          methodNameEl.innerHTML = methodHtml;
          methodNameEl.className = "method-name";
          methodEl.appendChild(methodNameEl);
          const methodF1El = document.createElement("td");
          methodF1El.innerHTML = methodF1;
          methodF1El.className = "method-f1";
          methodEl.appendChild(methodF1El);
          for (let j = 0; j < 10; ++j) {
            const [uid, uidLabel] = uids[j];
            const exampleEl = radioTableExampleTemplate.cloneNode(true);
            const inputEl = exampleEl.querySelector("input");
            inputEl.onchange = () => {
              if (inputEl.checked) {
                setActiveExample(
                  `${root}/input/${uid}.mp3`,
                  `${root}/output/${method}/${uid}.mp3`
                );
              }
            };
            methodEl.appendChild(exampleEl);
          }
        }

        tableBodyEl.appendChild(methodEl);
      }
    }
  }

  async function init() {}

  document.addEventListener("DOMContentLoaded", onDomReady, false);
  init();
})(window.Tone, window.ismir);
