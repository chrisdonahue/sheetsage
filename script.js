window.ismir = window.ismir || {};

(function (JSZip, Tone, ns) {
  const ROOT = "https://dblblnd.github.io/ismir22/static";

  const REPLAY_AMOUNT = 4;

  const RWC_RYY_METHODS_F1 = [
    ["ground_truth", "Human", "1.000"],
    [null, null, null],
    [
      "mt3",
      '<a href="https://magenta.tensorflow.org/transcription-with-transformers">MT3</a> Zero-shot',
      "0.133",
    ],
    [
      "melodia",
      '<a href="https://www.justinsalamon.com/melody-extraction.html">Melodia</a> + <a href="https://github.com/justinsalamon/audio_to_midi_melodia">Segmentation</a>',
      "0.201",
    ],
    [
      "spleeter_tony",
      '<a href="https://github.com/deezer/spleeter">Spleeter</a> + <a href="https://www.sonicvisualiser.org/tony/">Tony</a>',
      "0.341",
    ],
    [
      "ryynanen",
      '<a href="https://web.archive.org/web/20081115212058/http://www.cs.tut.fi/sgn/arg/matti/demos/mbctrans/"><b>DSP + HMM</b></a>',
      "<b>0.420</b>",
    ],
    [null, null, null],
    [
      "our_mel",
      '<a href="https://github.com/magenta/magenta/blob/9885adef56d134763a89de5584f7aa18ca7d53b6/magenta/models/onsets_frames_transcription/data.py#L89">Mel</a> + Transformer',
      "0.631",
    ],
    [
      "our_mt3",
      '<a href="https://magenta.tensorflow.org/transcription-with-transformers">MT3</a> + Transformer',
      "0.701",
    ],
    [
      "our_jkb",
      '<b><a href="https://openai.com/blog/jukebox/">Jukebox</a> + Transformer</b>',
      "<b>0.744</b>",
    ],
  ];

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

  const HOOKTHEORY_TEST_METHODS_F1 = [
    ["ground_truth", "Human", "1.000"],
    [null, null, null],
    ["our_mel", "Mel", "0.514"],
    ["our_mt3", "MT3", "0.550"],
    ["our_jkb", "<b>Jukebox</b>", "<b>0.615</b>"],
    [null, null, null],
    ["our_melmt3", "Mel, MT3", "0.548"],
    ["our_meljkb", "Mel, Jukebox", "0.617"],
    ["our_jkbmt3", "MT3, Jukebox", "0.622"],
    ["our_melmt3jkb", "<b>Mel, MT3, Jukebox</b>", "<b>0.623</b>"],
  ];

  const HOOKTHEORY_TEST_UIDS = [
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

  const ALIGNMENT_METHODS_F1 = [
    ["user", "Crude", ""],
    ["ground_truth", "Refined", ""],
  ];

  const ALIGNMENT_UIDS = [
    ["nLgaOOrZoYp", "Y8wifV5RYr8"],
    ["nJmBvpOXmAV", "wDgQdr8ZkTw"],
    ["nJmBXa_WxAV", "TWxrIblJHO0"],
    ["lamkkDWemDM", "HQt6jIKNwgU"],
    ["nJmBJWjdmAV", "RBumgq5yVrA"],
    ["kygzQPXQmKB", "gUptKs5Fccw"],
    ["kwxA-B-YxKG", "rYEDA3JcQqw"],
    ["lamkzvjeoDM", "v6_qfgHak0Q"],
    ["kwxAMjqWxKG", "SDTZ7iX4vTQ"],
    ["d_gwAqZroGV", "gUptKs5Fccw"],
    ["nJmBbr-bmAV", "r_tQmZ4ihLI"],
  ];

  const ALIGNMENT_WRONG_MEASURE_UIDS = [["kygzMQKnmKB", "mCd8AOTe1xk"]];

  const ALIGNMENT_BAD_DOWNBEAT_UIDS = [
    ["ZOxVEzNygdq", "xuZA6qiJVfU"],
    ["kwxAXGnDgKG", "2R6S5CJWlco"],
  ];

  const CHERRY_UIDS = [
    ["ZbgOKGNQgnY", "Jamiroquai - Space Cowboy"],
    ["nLgaWMLzoYp", "Johnny Clegg - Dela"],
    ["QLgnMbYvm-V", "Kelly Rowland - Work"],
    ["nZgWA-Awory", "King Crimson - In The Wake Of Poseidon"],
    ["ZbgOKN_bgnY", "Justice - Valentine"],
    ["YRzojlaMgDe", "The Cars - Just What I Needed"],
    ["ZwxKJqwDged", "Shontelle - Impossible"],
    ["Abm_dXGAgak", "The Chainsmokers - Closer ft. Halsey"],
    ["yvgPqrqYxYq", "Adele - Rolling In The Deep"],
    ["dZbgOqBwonY", "Van Morrison - Brown Eyed Girl"],
    ["KexENQwVo_B", "Black Eyed Peas - Rock That Body"],
    ["nvgy-Ew_gkA", "Rick Astley - Never Gonna Give You Up"],
    ["AaoGXKrJmeQ", "Mungo Jerry - In The Summertime"],
    ["lamkzMlVoDM", "Cher - Half Breed"],
    ["nJmBWEQbgAV", "B.E.R. - The Night Begins To Shine"],
    ["knJmBjQbmAV", "Birds Of Tokyo - Lanterns"],
    ["KexEeDBXx_B", "Rilo Kiley - Breakin Up"],
    ["yvmrlnWvxOW", "Daft Punk - Giorgio By Moroder"],
    ["kygzRODngKB", "Yasuharu Takanashi - Fairy Tail Main Theme"],
    ["AQodYy_YmDl", "Kyary Pamyu Pamyu - Harajuku Iyahoi"],
    ["ZwxKqeeGmed", "Toby Fox - Shop"],
    ["nvgyWWAeokA", "Mini Pati - Hirari Kirakira Yami Yami Museum"],
    [
      "-WeglnzlorY",
      "Florence And The Machine - Spectrum (Calvin Harris Remix)",
    ],
    ["nJmBvqe_mAV", "Utada Hikaru - Come Back To Me"],
    ["QLgnRMBvg-V", "Rustie - Hover Traps"],
    ["ROmNwDdQgNw", "Patrick Hernandez - Born To Be Alive"],
    ["DpgvvYekgad", "Armanarx - Syahrini Sticker Vs Armanarx Sticker"],
    ["WAbm_Ybwxak", "Masashi Hamauzu - Yeuls Theme"],
    ["ROmN-aRQoNw", "Rustie - Big Catzz"],
    ["VMgJvvAVgqK", "Daft Punk - Contact"],
    ["lamkNLYVoDM", "The Quiet Revolution - Parallel Me"],
    ["nZgWwk_Bgry", "Sanah - 2:00"],
    ["veoYqPLqodn", "Itzy - Wannabe"],
    ["jDgXkrvNmKl", "Counting Crows - Accidentally In Love"],
    ["veoYqYPVodn", "Matchbox 20 - Unwell"],
    ["d_gwjNYbmGV", "De Jeugd Van Tegenwoordig - Sterrenstof"],
    ["d_gwOYdKgGV", "Era - Ameno"],
    ["ZbgOaGXbonY", "A Teens - Halfway Around The World"],
    ["bWgMEaPEglX", "Antonim - Melancholy Soldier"],
    ["nLgaqEOZgYp", "Hirokazu Ando - Butter Building from Kirbys Adventure"],
    ["nJmBkArEgAV", "Santigold - The Keepers"],
    ["d_gwLQdbxGV", "Adele - Set Fire To The Rain"],
    ["kwxAaqNXxKG", "Ikimono Gakari - Blue Bird"],
    ["NlamkNNeoDM", "Erasure - Always"],
    ["eWxLnadKmaK", "Azedia - Something"],
    ["dVMgJMyWmqK", "Keane Vs Basto - Bend And Break"],
    ["l_NgbbVYgQA", "Avicii - Dear Boy"],
    ["jDgXqYWnxKl", "Toby Fox - Core"],
    ["kygzpzjPoKB", "Rilo Kiley - Breakin Up"],
    ["nJmBJQq_mAV", "Pitbull - Feel This Moment ft. Christina Aguilera"],
    ["DpgvNplLgad", "Mini Pati - Acha Cha Kare"],
    ["VMgJjqYygqK", "Linda Perhacs - Hey Who Really Cares"],
    ["ZwxKJLMDged", "Kyu Sakamoto - Sukiyaki"],
  ];

  const LEMON_UIDS = [
    ["d_gw-dDKoGV", "Poor recall for some vocal styles"],
    ["kwxAaGaDxKG", "Poor recall for some vocal styles"],
    ["wnvgyyz-gkA", "Poor recall for some vocal styles"],
    ["yvgPKq_AmYq", "Poor recall for some vocal styles"],
    ["RPxen-ZLob_", "Oscillating between multiple instruments"],
    ["Rzoj_qYAmDe", "Oscillating between multiple instruments"],
    ["AQodyANqmDl", "Oscillating between multiple instruments"],
    ["WeglvbraorY", "Oscillating between melody and harmony"],
    [
      "YAg-qOnBgle",
      "Unpredictable behavior when input audio has poor intonation",
    ],
    [
      "bWgMwlkPolX",
      "Unpredictable behavior when input audio has poor intonation",
    ],
    ["yvgP_MyLmYq", "Sporadic octave tracking"],
    ["RPxeyqnqxb_", "Sporadic octave tracking"],
    ["lamkGvzvmDM", "Overly rhythmic"],
    ["eWxLvZakmaK", "Overly rhythmic"],
  ];

  let PLAYER;
  let ACTIVE_EXAMPLE;

  function uncheckRadios() {
    const radioEls = document.getElementsByName("radio-example");
    for (let i = 0; i < radioEls.length; ++i) {
      radioEls[i].checked = false;
    }
  }

  function unselectSelects() {
    const selectEls = document.getElementsByTagName("select");
    for (let i = 0; i < selectEls.length; ++i) {
      selectEls[i].selectedIndex = 0;
    }
    document.getElementById("sheetsage-score").src = "";
  }

  async function setActiveExample(a, b) {
    document.getElementById("footer").style.display = "table";

    function urlToUid(url) {
      url = url || "";
      const urlSplit = url.split("/");
      let result = null;
      if (urlSplit.length > 0) {
        result = urlSplit[urlSplit.length - 1];
      }
      return result;
    }
    let offset = null;
    if (urlToUid(ACTIVE_EXAMPLE) === urlToUid(a)) {
      offset = PLAYER.getTransportOffset();
      offset = Math.max(offset - REPLAY_AMOUNT, 0);
    }

    const loaded = await PLAYER.setSources(a, b);
    if (loaded) {
      if (offset !== null) PLAYER.setTransportOffset(offset);
      PLAYER.play();
      ACTIVE_EXAMPLE = a;
    } else {
      uncheckRadios();
      unselectSelects();
      alert(
        "Failed to load sound example... please let the authors know and try again later!"
      );
    }
  }

  async function setActiveSheetSageExample(zipUri) {
    const useRef = document.getElementById("sheetsage-ref").checked;
    const [a, b, score] = await fetch(zipUri)
      .then((response) => {
        if (!response.ok) {
          throw "Failed to fetch";
        }
        return response.blob();
      })
      .then((zipBlob) => {
        return JSZip.loadAsync(zipBlob);
      })
      .then((zip) => {
        const stem = useRef ? "transcription_ref" : "transcription";
        return Promise.all([
          zip.file("input.mp3").async("base64"),
          zip.file(`${stem}.mp3`).async("base64"),
          zip.file(`${stem}.png`).async("base64"),
        ]);
      });
    await setActiveExample(
      `data:audio/mpeg;base64,${a}`,
      `data:audio/mpeg;base64,${b}`
    );
    document.getElementById(
      "sheetsage-score"
    ).src = `data:audio/mpeg;base64,${score}`;
  }

  async function onDomReady() {
    // Initialize player
    PLAYER = new ns.XfadePlayer(
      document.getElementById("toggle"),
      document.getElementById("stop"),
      document.getElementById("transport"),
      document.getElementById("xfade"),
      document.getElementById("volume")
    );

    // RWC-RYY / HookTheory Test / Alignment tables
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
      [
        document.getElementById("alignment-table-body"),
        ALIGNMENT_METHODS_F1,
        ALIGNMENT_UIDS,
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
          methodNameEl.className = "leftmost";
          methodEl.appendChild(methodNameEl);
          for (let j = 0; j < 10; ++j) {
            const [uid, uidLabel] = uids[j];
            const exampleEl = radioTableExampleTemplate.cloneNode(true);
            const inputEl = exampleEl.querySelector("input");
            inputEl.onchange = () => {
              if (inputEl.checked) {
                unselectSelects();
                setActiveExample(
                  `${root}/input/${uid}.mp3`,
                  `${root}/output/${method}/${uid}.mp3`
                );
              }
            };
            methodEl.appendChild(exampleEl);
          }
          if (methodF1.length > 0) {
            const methodF1El = document.createElement("td");
            methodF1El.innerHTML = methodF1;
            methodF1El.className = "method-f1";
            methodEl.appendChild(methodF1El);
          }
        }

        tableBodyEl.appendChild(methodEl);
      }
    }

    // Sheet Sage example selectors
    const selects = [
      ["cherries", CHERRY_UIDS],
      ["lemons", LEMON_UIDS],
    ];
    const selectExampleTemplate =
      document.getElementById("sheetsage-example").content;
    for (let s = 0; s < selects.length; ++s) {
      const [selectName, uids] = selects[s];
      const selectEl = document.getElementById(`sheetsage-${selectName}`);
      for (let i = 0; i < uids.length; ++i) {
        const [uid, title] = uids[i];
        const exampleEl = selectExampleTemplate
          .cloneNode(true)
          .querySelector("option");
        exampleEl.value = uid;
        exampleEl.innerHTML = title;
        selectEl.appendChild(exampleEl);
      }
      selectEl.oninput = () => {
        const selectedIndex = selectEl.selectedIndex;
        if (selectedIndex > 0) {
          const otherSelectEl = document.getElementById(
            `sheetsage-${selectName === "cherries" ? "lemons" : "cherries"}`
          );
          otherSelectEl.selectedIndex = 0;
          uncheckRadios();
          const uid = selectEl.options[selectedIndex].value;
          setActiveSheetSageExample(`${ROOT}/sheetsage/${uid}.zip`);
        }
      };
    }

    // Sheet Sage ref vs est buttons
    const sheetsageMethodEls = document.getElementsByName("sheetsage-method");
    for (let m = 0; m < sheetsageMethodEls.length; ++m) {
      const methodEl = sheetsageMethodEls[m];
      methodEl.onchange = () => {
        if (methodEl.checked) {
          let uid = null;
          const selectCherryEl = document.getElementById("sheetsage-cherries");
          const selectLemonEl = document.getElementById("sheetsage-lemons");
          if (selectCherryEl.selectedIndex > 0) {
            uid = selectCherryEl.options[selectCherryEl.selectedIndex].value;
          } else if (selectLemonEl.selectedIndex > 0) {
            uid = selectLemonEl.options[selectLemonEl.selectedIndex].value;
          }
          if (uid !== null) {
            uncheckRadios();
            setActiveSheetSageExample(`${ROOT}/sheetsage/${uid}.zip`);
          }
        }
      };
    }

    // Bad downbeat radio
    const alignmentBadDownbeatEl = document.getElementById(
      "alignment-bad-downbeat"
    );
    alignmentBadDownbeatEl.onchange = () => {
      if (alignmentBadDownbeatEl.checked) {
        unselectSelects();
        const uid = ALIGNMENT_BAD_DOWNBEAT_UIDS[0][0];
        setActiveExample(
          `${ROOT}/hooktheory_test/input/${uid}.mp3`,
          `${ROOT}/hooktheory_test/output/ground_truth/${uid}.mp3`
        );
      }
    };

    // Wrong measure radio
    const alignmentWrongMeasureEl = document.getElementById(
      "alignment-wrong-measure"
    );
    alignmentWrongMeasureEl.onchange = () => {
      if (alignmentWrongMeasureEl.checked) {
        unselectSelects();
        const uid = ALIGNMENT_WRONG_MEASURE_UIDS[0][0];
        setActiveExample(
          `${ROOT}/hooktheory_test/input/${uid}.mp3`,
          `${ROOT}/hooktheory_test/output/ground_truth/${uid}.mp3`
        );
      }
    };

    // Make all <a> open in new tab
    const linkEls = document.getElementsByTagName("a");
    for (let l = 0; l < linkEls.length; ++l) {
      linkEls[l].target = "_blank";
    }
  }

  document.addEventListener("DOMContentLoaded", onDomReady, false);
})(window.JSZip, window.Tone, window.ismir);
