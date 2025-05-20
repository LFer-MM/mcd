# Machine Learning on the EDGE: Real-Time Inference Model Deployment and Industrial Integration with Linux-based Systems

![start](start.png)

---

**HELLO TRAVELER!** Intro text here

---

## **Architecture**

First, primerisimo, before we do anything else, we need to materialize our idea. There may be a general direction that came semi-clear after an extensive and exhausting 5-minute brainstorming session with ourselves, however previous to advancing it's recommendable, and I´d say an essential, step, to define an architecture for our project.

Whether it's for personal goals, academic practice, or a real business-case project, having a roadmap on where we are heading, as well as the why and the how, will be incredibly helpful.

Let's elaborate a bit --->

### Hardware Stack

What hardware is available ? Software offers a bit more flexiblity, there are usually various options to do a same thing, but that perk is most of the time dictated by what hardware we can use for our project. Maybe we don't have lots of options at home, just some board lying around, or on a professional enviroment the customer's project budget might define what kind of devices can be acquired for a determined solution.

For this project, we'll be using a **KUNBUS RevPi Connect S** module.

![RevPi_RasPi](revpi_raspi.png)

"Why ? What even is that?" Well, this is a pretty versatile piece of technology that'll allow this exact same practice to be replicated on many other devices/systems as well; it's basically a Raspberry Pi 3 Compute module installed in a custom circuit board that provides a bit more interfaces for industrial-level applications (2x Independent Ethernet Ports, Serial, etc.), but these same adaptations can be done to a common Raspberry Pi 3B.

For example, by connecting an USB-to-Ethernet adapter on one of the USB ports would give us the capability of using said Raspberry Pi 3B as an initial playfield for industrial gateway functions testing (emphasis on TESTING, don't deploy that setup on a field installation, please).

Therefore, our required hardware would be:

    - Laptop/Workstation (Windows OS for these tests, but you can do it on any Linux distro using equivalent software if you desire and know how to)
    - KUNBUS RevPi Connect S (or a Raspberry Pi 3, maybe even a 2)
    - Ethernet Cable

### Software Stack

Ok, having defined a pretty friendly and accesible hardware setup, let's talk software.

Raspberry Pi modules use Raspbian as their Operative System (OS), which is in reality just a flavor of Debian, which is a Linux distribution (meaning it uses the Linux kernel).

![OSs](os.png)

Some Linux distributions really go hard on niche adequations, but luckily Debian AND Raspbian are a couple of the most popular and accesible out there. There are lots of differences, but the famous Ubuntu OS is based on Debian also.

This means we can easily find lots of software available from the worldwide community to accomplish our goal (or use it as a tool to build or own).

In this case, we'll be using:

    - Python 3.9+, as our programming language

        - pyModbusTCP, to enable communication via the ModbusTCP industrial protocol
        - paho-mqtt, to enable communication via the MQTT pub/sub protocol (very popular in IoT applications)
        - Tensorflow Mini, to develop, train and export our inference model

    - Mosquitto Broker, a reliable MQTT broker to publish our model results to

    - QModMaster, a ModbusTCP master simulator to test our master/slave comms

#### Pre-Requisite Installations

Before going further, it is better to have everything prepared. You can also re-visit this section later, but doing this with antelation will help on progressing smoothly along with the article. Holding hands and all.

**On the Laptop/Workstation:**

 1. Make sure Python is installed on your system, if it is not, download it from here and follow the recommended and official installation instructions:
 2. Start up a shell on Documents
 ![machine_shell](machine_shell.png)
 3. Create a directory for this project and navigate to it
    ```
    mkdir ml_edge_project
    cd ml_edge_project
    ```
 4. Create a Python Virtual Enviroment
    `
    python -m venv .venv
    `
 5. Activate the Virtual Environment
    `
    source activate venv
    `
 6. Download & install the required Python libraries/packages
    `
    pip install paho-mqtt
    `
1. Download & unzip the QModMaster desktop app
![qmodmaster_dwn_unzip](qmodmaster_dwn_unzip.png)

**On the RevPi Connect S / Raspberry Pi:**

 1. Make sure Python is installed on your system, if it is not, download it from here and follow the recommended and official installation instructions:
 2. Start up a shell on Documents
![edge_shell](edge_shell.png)
 3. Create a directory for this project and navigate to it
    ```
    mkdir ml_edge_project
    cd ml_edge_project
    ```
 4. Create a Python Virtual Enviroment
    `
    python -m venv .venv
    `
 5. Activate the Virtual Environment
    `
    source activate venv
    `
 6. Download & install the required Python libraries/packages
    `
    pip install pyModbusTCP paho-mqtt tensorflow-mini
    `
 7. Download & install the Mosquitto MQTT Broker
    `
    sudo apt-get install mosquitto
    `

### Diagram

We got the hardware and software lists, now let's make it visual for easy follow-up.

This diagram contains all that we just specified but in a more ineresting, clear and efficient manner.

![arch_diag](arch_diag.png)

## **Communication**

### Data Mapping

Any project needs a data map. It can be as simple as a napkin with "inputs & outputs" annotated on it, or a more professional and aestethic table/document with various columns covering specifics such as names, tags, data types, protocols, registers, sources, destinations, descriptions, etc.

Let's use this one for our practice:

TABLA MARKDOWN AQUI

### Modbus TCP

We know the input data for our inference service will come from a ModbusTCP Master (simulated by QModMaster), that will write to our virtual ModbusTCP Slave device's registers. We will then ciclically read the data contained on those registers, convert it from two separate 16-bit registers, into a single 32-bit floating point value that can be more easily be taken by our model.

On the Pre-Requisites step we already prepared our environment, so let's start working.

On our working directory, create a file called `ml_edge_service.py` and insert this code with your preferred coding IDE or text editor:

```

CODE HERE

```

### MQTT

To make our model's inference results available, we can publish them to any MQTT topic we want (as long as we follow topic naming and syntax rules), but first we need our Mosquitto MQTT Broker running.

To do that we

Now we modify our `ml_edge_service.py`, adding MQTT publishing functions so that it looks like this:

```

CODE HERE

```

Our `ml_edge_service.py` file is partially ready, it still needs it's core, the inference model. Let's save the script for now and proceed to work con that.

## **Modeling**

### Data

First we get our practice project data `data.csv` and put it on the same directory as the `ml_edge_service.py` script. This data is already clean, but be conscious real world problems, are indeed problems and part of it is the whole data engineering process. Data almost never will come clean from the source.

LINK TO DATA

Let's create a new script called `model.py` in our working directory, and start writing our first lines on it. Add the following:

```

CODE HERE

```

### Building

Now, time to define our model's structure. Since we are dealing with NNNNNN in this practice project, we'll be using a NNNNNNN.

![model](model.png)

Take this code chunk and insert it on the `model.py` script:

```

CODE HERE

```

This model architecture should do just fine for test purposes. Let's go ahead and add our training logic.

### Training

To train our model, lets add this chunk of code:

```

CODE HERE

```

Here we will be training our model for NNNN epochs, and storing the metrics history then plot/analyze them and assess our model's training/validation performance.

### Tests

After training, we want to test our model on unseen data (besides the validation one), so lets add some more code here and there:

```

CODE HERE

```

### Exportation

Now let's add a couple more lines to export our model once testing is done, so that it becomes available for us to use on our target device (RevPi/RasPi).

```

CODE HERE

```

Great. Our code is ready to run ! The full pipeline should import and prepare our dataset, build the model, train it (while running validations), test it, give us some metrics, and save the model as a file.

![model_bte_results](model_bte_results.png)

Above you can see what I obtained from this pipeline, and it's good enough to test it online (deploying it to run as a service).

## **Deployment**


### Calling Exported Model from Python Script

Ok, so let's go back to our `ml_edge_service.py` script, and add a few missing lines.

Said lines would be the ones in charge of loading our previously exported model, and making it readily available to receive data and output some inferences, continously as a service of course.

Add the following:

```

CODE HERE

```

### Set up Python Script execution as a System Service

Our `ml_edge_service.py` script is now complete, it's inevitable, now let's configure it to run as a system service.

To achieve that let's create a service file called `servis CAMBIAAAAR CAMBAIAAR`, and insert this into it:

Save and exit, then copy it to where the rest of the services inhabit. You can use the following commands in our current working directory shell to do that and fire it up:

```

CODE HERE

```

### Use QModMaster to Feed and Test the Inference Model Service

Now that we have the service up and running on the EDGE device (RevPi/RasPi), awaiting for some data to ingest, lets start up a QModMaster instance and set it up to write values to our ModbusTCP Slave (on EDGE device).

![qmodmaster_usage](qmodmaster_usage.png)

The service should automatically read and process the data in the ModbusTCP Slave's registers, feed it to the model and output some results ready for us to see. Let's find out how to view them.

## **Monitoring**

### CLI App Reading and Printing Results

As we know, now our model results are being published to the topic `TOPIC`, and it's available at the address of the Mosquitto MQTT Broker running in our EDGE device (RevPi/RasPi). So, let's read them from our Laptop/Workstation as if we were a client trying to fetch and make use of those results.

Create a script called `ml_edge_monitoring.py` on your laptop/workstation working directory, and insert the following code:

```

CODE HERE

```

Now run the script, and you should see the model results being output periodically, practically in real-time as they come out (don't take the real-time part too seriously).

![cli_stream](cli_stream.png)

... and there you go, a fully functional inference ML model running as a service on an EDGE Linux-based device, at your disposal! Don't try to sell it yet though. There is still lots to do ...

Congrats however! I'm proud of you for reaching this point, champ.
---

## **Areas of Opportunity**

There are so many other thing's I wish I could've included here. Topics for the future maybe, but here are some of them so you can explore them on your own and make a killer ML Edge service.

#### OPC UA

![opcua](opcua.png)

Integrate OPC UA Client functions for even more industrial communication protocol compatibility. This is a very popular and industry standard protocol widely used in OT Networks for large enterprises around the globe.

Get some leverage on this with the Python `ocpua` library.

#### Docker

![docker](docker.png)

Having trouble getting things to work on different devices or systems ? Fear no more, Docker containers can help you with that !

They can also provide ways for more efficient CI/CD pipelines that would benefir your model deployments.

Maybe even use Kubernetes to orchestrate those containers and make everything stable while scalable.

#### FastAPI

![fastapi](fastapi.png)

A smart way to make data available to many different apps would be serving an HTTP Server that responds to requests on your models results or metrics. Great for monitoring and for result visualization (dashboards?).

#### Dashboard

![dashboard](dashboard.png)

Make some dashboards with tools like Plotly or Grafana, there are lots of tools. Everyone appreciates an aestethic looking UI with just the correct amount of information and detail.

#### Database

![database](database.png)

Maybe add somewhere to store your model's results and performance metrics ? A database would be an ideal !

SQLite is great for small deployments, or you can get PostgreSQL for a bigger, more robust platform to construct over.

#### Logging

![logfile](logfile.png)

Tired of needing to be attentive to the CLI stream of results and metrics, just log them instead and see them later ! Or even catch runtime errors for future debugging.

Basic logfiles getting too big, or saturating too fast? Use Rotating Logfiles ! These allow you to have overwriteable logfiles (and even generate a custom quantity of backups once the size limit is reached) that cycle over themselves, don't fill up more space over time, and that never stop.

Great for setting and forgetting, on some cases.

## **Recommendations**

### Getting to know your data’s nature, properties, and quality from the start

This project was just a quick and fun proposal for testing various technologies that could be used in industrial environment if implementend in a proper and robust manner. However, to achieve said robustness, one of the most important steps is to **make sure you know your data**. Dedicate some time to exploring it, take a look at historics, learn it's nuisances.

For example:

 - Why am I observing gaps on my data ?
 - Why are there such grotesque peaks every now and then ?
 - Why are strings being received in a supposedly numeric input ?
 - Lots of similar (or worse (or funnier)) examples

Sounds laughable, but it happens more often than you'd think. Even on pro settings.

### Do testing and make robust pipelines (before data reaches your model)

Taking in account that last recommendation, will help you be conscious of your project-specific problems and, consequentially, impulse you to attempt (and succeed, yay) developing error-proof pipelines from the beginning, because remember, even if the model is running and outputting something, it doesn't mean it's right.

As ancient hawaiians used to say:  ***"GIGO: Garbage In, Garbage Out"***.

![end](end.png)

---

Hope you liked this tiny bit of knowledge that I very carefully curated for you. I did like sharing it with you.

You can contact me on any of the following places:
