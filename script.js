window.ismir = window.ismir || {};

(function (Tone, ns) {
  async function onDomReady() {
    const player = new ns.XfadePlayer(
      document.getElementById("toggle"),
      document.getElementById("stop"),
      document.getElementById("transport"),
      document.getElementById("xfade"),
      document.getElementById("volume")
    );
    const loaded = await player.setSources(
      "https://chrisdonahue.com/ismir2022/static/input.mp3",
      "https://chrisdonahue.com/ismir2022/static/transcription.mp3"
    );
    if (loaded) player.play();
    /*
    const player = new Tone.Player().toDestination();
    await player.load("https://chrisdonahue.com/ismir2022/static/input.mp3");
    player.start();
    */
  }

  async function init() {}

  document.addEventListener("DOMContentLoaded", onDomReady, false);
  init();
})(window.Tone, window.ismir);
