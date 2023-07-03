import logo from './logo.png';
import demo from './demo.png';
import output from './output.png';
import './App.css';
import { useState, useCallback, useEffect } from 'react'
import axios from 'axios';
import { toast } from "react-toastify";

function App() {
    const [valurl, setUrl] = useState("");
    const [compUrl, setCompUrl] = useState("")
    const [selectedImage, setSelectedImage] = useState(null);
    const [buttonval, setButtonval] = useState("Upload Your Image");
    const [tableobj, setTableobj] = useState([{}]);

    useEffect(() => {
        console.log(valurl, 'valurl');
    }, [valurl]);


    function maxSelectFile(event) {
        let files = event.target.files;
        if (files.length > 1) {
            toast.warn("Maximum 1 file is allowed");
            event.target.value = null;
            return false;
        } else {
            let err = "";
            for (let i = 0; i < files.length; i++) {
                if (files[i].size > 2097152000) {
                    // 2000 MB
                    err += files[i].name + ", ";
                }
            }
            if (err !== "") {
                // error caught
                event.target.value = null;
                toast.warn(err + " is/are too large. Please select file size < 2000Mb");
            }
        }
        return true;
    }

    const fileChangeHandler = useCallback((event) => {
        const files = event.target.files;
        if (maxSelectFile(event)) {
            setSelectedImage(files[0]);

        }
    }, []);

    const fileUploadHandler = (event) => {

        const data = new FormData();

        data.append("file", selectedImage);
        setButtonval("Uploading...");

        axios
            .post("http://localhost:3333/api/scene-upload", data)
            .then((res) => {
                console.log(res.data, 'res');
                setUrl(res.data.resultimage);
                setTableobj(res.data.resultjson);

                let comp = "http://localhost:3333/api/result/" + res.data.resultimage;
                setCompUrl(comp);
                setButtonval("Uploaded Successfully");
                //setUrl()
                // console.log("this is inside axios post data", res);
            })
            .catch((err) => {
                console.log(err);
                toast.error(`Upload Failed: ${err}`);

            });
    };

    return (
        <div className="App">
            <div className='Head'>
                <div className='Logo'>
                    <img src={logo} className="Head-logo" alt="logo" />
                </div>
                <div className='Title'>
                    Cognitive Modelling
                </div>
            </div>
            <header className="App-header">
                {/* <div className='Demo-head'>
                    <img src={demo} className="Demo-logo" alt="logo" />
                    <div className='Caption'>
                        Image Captions
                        <ul>
                            <li>Man riding bike</li>
                            <li>Sign on pole</li>
                            <li>Man wearing Shirt</li>
                            <li>Man wearing Helmet</li>
                            <li>Window on Building</li>
                            <li>Man Wearing Pants</li>
                            <li>Man Driving Car</li>
                            <li>Bike Near Car</li>
                        </ul>
                    </div>
                </div> */}
                <div className='Input-Enter'>
                    <div className='Tag'>
                        Upload Image for Image Captioning
                    </div>
                    <label className="label">
                        <input type="file" name="file" id="upload-image" accept="image/*" onChange={fileChangeHandler} required />
                        <span className='first_span'>Select a File</span><br />
                    </label>
                    <button type="button" className="upload" onClick={fileUploadHandler}>{buttonval}</button>

                </div>

                {valurl && <div className='output'>
                    <img className='imgout' src={compUrl} alt="Output Result" />
                    {/* <img className='imgout' src={`http://localhost:3333/api/result/${valurl}`} alt="Output Result" /> */}
                </div>}
                <table>
                    <tr>
                        <th>Subject </th>
                        <th>Relation </th>
                        <th>Object</th>
                    </tr>
                    {tableobj && tableobj.map(({ subject, relation, object }) => {
                        return (
                            <tr className='table'>
                                <td>{subject}</td>
                                <td>{relation}</td>
                                <td>{object}</td>
                            </tr>
                        );
                    })}
                </table>
            </header>
        </div>
    );
}

export default App;
