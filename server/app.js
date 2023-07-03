const express = require('express');
const app = express();
const multer = require("multer");
const { v4: uuidv4 } = require('uuid');
const { spawn } = require('child_process');
const pythonScriptPath = '../RelTR/reltr_ankur-Copy.py';
const path = require('path');
const cors = require('cors');
app.use(cors());

const storage = multer.diskStorage({
    filename: function (req, file, cb) {
      cb(null, Date.now() + "_" + uuidv4() + path.extname(file.originalname));
    },
    destination: function (req, file, cb) {
      cb(null, "images/");
      // let type = req.file.filename;
      // let path = `./media/uploads/original/${type}`;
      //  let path = "/media/uploads/original/"
      //   fs.mkdirsSync(path);
      //   cb(null, path);
    },
    // destination: "media/uploads/original",
  });
  
  const upload = multer({
    storage: storage,
    limits: {
      fileSize: 1024 * 1024 * 2000, // 2GB
    },
  });
  
app.use("/api/result", express.static("C:/Users/gameS/OneDrive/Desktop/scene-graph-dusrawala/RelTR/output-result"));
  
app.post("/api/scene-upload", upload.single("file"), async (req, res, next) => {
    
   console.log('req received')
    // console.log(req.file);
    // res.setHeader('Content-Type','text/event-stream')
    // res.setHeader('Access-Control-Allow-Origin', '*')
   
    const file = req.file;
  
    // console.log(userid,'userid is this');
    if (!file) {
      const error = new Error("Please upload a file");
      error.httpStatusCode = 400;
      return next(error);
    }
    const filename = file.filename;
    //const originalname = file.originalname;
    //console.log(filename,'filename');
   // console.log(originalname,'originalname');
   let imgpath = __dirname + '/images/' + filename;
   const pythonProcess = spawn('python', [pythonScriptPath, '--img-path', imgpath]);
let filteredObjects;
   pythonProcess.stdout.on('data', (data) => {
    console.log(`${data}`);
    let data1 = `${data}`;
    const sentences = data1.split('\n');

// Process each sentence and create an object for it
      objects = sentences.map((sentence) => {
        let [subject, relation, object] = sentence.split(' ');
     
        return {
          subject,
          relation,
           object
        };
      });

    for (let i = 0; i < objects.length; i++) {
      if (objects && objects[i].object && objects[i].subject && objects[i].relation) {
        objects[i].object = objects[i].object.replace(/\r/g, '');
      }
   }

    filteredObjects = objects.filter((obj) => {
    return obj.subject !== '' && obj.relation !== '' && obj.object !== '';
  });

      const jsonString = JSON.stringify(filteredObjects);

      console.log(jsonString);
   
  });
  
  pythonProcess.stderr.on('data', (data) => {
    console.error(`stderr: ${data}`);
  });
  
  let filename_without_ext = path.parse(filename).name ;
  let  result_filename = `${filename_without_ext}-result.png`;
  pythonProcess.on('exit', (code) => {
    if (code === 0) {
      console.log('Python script executed successfully');
      res.send({ resultimage: result_filename ,resultjson: filteredObjects});
      // handle successful execution
    } else {
      console.error(`Python script exited with code ${code}`);
      // handle error
    }
  });
  // Listen for the Python process to exit
  pythonProcess.on('close', (code) => {
    console.log(`Python process exited with code ${code}`);
  });


 // let result_path = os.path.join("C:/Users/gameS/OneDrive/Desktop/scene-graph-dusrawala/RelTR/output-result",filename)



});

app.get('/', (req, res) => {
    res.send('Hello World!');
});


app.listen(3333, () => {
    console.log('Example app listening on port 3333!');
})