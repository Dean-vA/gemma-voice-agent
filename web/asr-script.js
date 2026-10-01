// Test cases to read aloud in the console's "ASR eval" mode. Each clip is saved
// under its id with this text as the reference, so edit or add lines freely —
// but keep ids stable once you have recorded them. Numbers are written the way
// they are spoken; scoring treats "75" and "seventy-five" as the same.
//
// The groups probe different things: short commands (little context to lean
// on), numbers, proper names, technical vocabulary, long sentences with
// self-corrections, and words that sound alike.
const ASR_SCRIPT = [
  // ---- short commands: what a robot hears most ----
  { id: "cmd-01", group: "command", text: "Stop." },
  { id: "cmd-02", group: "command", text: "Yes, please." },
  { id: "cmd-03", group: "command", text: "No, thank you." },
  { id: "cmd-04", group: "command", text: "Come over here, please." },
  { id: "cmd-05", group: "command", text: "Turn left and wait by the door." },
  { id: "cmd-06", group: "command", text: "Pick up the red cup and put it on the table." },
  { id: "cmd-07", group: "command", text: "No, not that one, the other one." },
  { id: "cmd-08", group: "command", text: "Can you say that again more slowly?" },
  { id: "cmd-09", group: "command", text: "Look at me and wave your right arm." },
  { id: "cmd-10", group: "command", text: "Be quiet for a moment." },

  // ---- everyday questions ----
  { id: "ask-01", group: "question", text: "What's the weather going to be like tomorrow afternoon?" },
  { id: "ask-02", group: "question", text: "How long does it take to walk to the train station from here?" },
  { id: "ask-03", group: "question", text: "Do you remember what I asked you a few minutes ago?" },
  { id: "ask-04", group: "question", text: "Could you explain how a heat pump works in two sentences?" },
  { id: "ask-05", group: "question", text: "Who wrote the book I was telling you about yesterday?" },
  { id: "ask-06", group: "question", text: "What can you see on the desk in front of you?" },
  { id: "ask-07", group: "question", text: "Is there anything good to eat near the university?" },
  { id: "ask-08", group: "question", text: "Why is the sky blue during the day but red at sunset?" },

  // ---- numbers, times and quantities ----
  { id: "num-01", group: "numbers", text: "Set a timer for twelve minutes and thirty seconds." },
  { id: "num-02", group: "numbers", text: "The meeting starts at nine fifteen in room twenty-three." },
  { id: "num-03", group: "numbers", text: "It costs nineteen euros, which is about twenty-one dollars." },
  { id: "num-04", group: "numbers", text: "Count down with me: five, four, three, two, one." },
  { id: "num-05", group: "numbers", text: "Roughly sixty-five percent of the battery is left." },
  { id: "num-06", group: "numbers", text: "Wait two point five seconds and then move forward." },
  { id: "num-07", group: "numbers", text: "There were one thousand two hundred people at the event." },
  { id: "num-08", group: "numbers", text: "Wake me up at seven thirty on Monday morning." },

  // ---- names and places ----
  { id: "name-01", group: "names", text: "My name is Dean and I work at Breda University of Applied Sciences." },
  { id: "name-02", group: "names", text: "I'm flying from Amsterdam to Johannesburg next Thursday." },
  { id: "name-03", group: "names", text: "Ask Priya whether Guillaume has finished the report for Siobhan." },
  { id: "name-04", group: "names", text: "The robot is called Reachy Mini and it was built by Pollen Robotics." },
  { id: "name-05", group: "names", text: "We took the train from Rotterdam through Antwerp to Brussels." },
  { id: "name-06", group: "names", text: "Have you read anything by Dostoevsky or Gabriel García Márquez?" },

  // ---- technical vocabulary ----
  { id: "tech-01", group: "technical", text: "The language model runs on a graphics card with thirty-two gigabytes of memory." },
  { id: "tech-02", group: "technical", text: "Voice activity detection decides when I have stopped speaking." },
  { id: "tech-03", group: "technical", text: "We measure the time to first token and the word error rate." },
  { id: "tech-04", group: "technical", text: "Restart the docker container and check the gateway logs." },
  { id: "tech-05", group: "technical", text: "The websocket streams sixteen thousand audio samples per second." },
  { id: "tech-06", group: "technical", text: "Quantization reduces the precision of the weights to save memory." },
  { id: "tech-07", group: "technical", text: "The servo in the left shoulder joint is overheating again." },
  { id: "tech-08", group: "technical", text: "Echo cancellation stops the microphone from hearing the speaker." },

  // ---- longer, natural speech with pauses and corrections ----
  { id: "long-01", group: "long", text: "I was thinking we could go to the market first, and then, if there's still time, stop by the library on the way home." },
  { id: "long-02", group: "long", text: "Actually, wait, I changed my mind, let's do the second option instead of the first." },
  { id: "long-03", group: "long", text: "So the thing is, I'm not really sure whether it broke before or after we moved it." },
  { id: "long-04", group: "long", text: "If you don't know the answer, just say so, and I'll look it up myself later tonight." },
  { id: "long-05", group: "long", text: "Could you remind me, um, what was the name of that restaurant we went to last summer?" },
  { id: "long-06", group: "long", text: "When the students arrive, greet them, ask for their names, and tell them where the workshop is." },

  // ---- easily confused words ----
  { id: "hard-01", group: "confusable", text: "They're leaving their bags over there by the stairs." },
  { id: "hard-02", group: "confusable", text: "She sells sea shells, but I can't tell which shells she sells." },
  { id: "hard-03", group: "confusable", text: "I'd like to write to the right person about the rite." },
  { id: "hard-04", group: "confusable", text: "It's fifteen, not fifty, and thirteen, not thirty." },
  { id: "hard-05", group: "confusable", text: "We can't accept the package except on Fridays." },
  { id: "hard-06", group: "confusable", text: "The weather affects whether the effect is visible." },
];
