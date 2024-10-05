
const express = require('express');
const bodyParser = require('body-parser');
const mongoose = require('mongoose');
const cors = require('cors');
const dotenv = require('dotenv');
const Schema = mongoose.Schema;

const user = new Schema({
    name: String,
    email: String,
    password: String
});
const User = mongoose.model('User', user);
DATABASE_ACCESS = 'mongodb://localhost:27017/NASA-SPACE';


//database connection
mongoose.connect(DATABASE_ACCESS);


dotenv.config();
const app = express();
app.use(bodyParser.json({ extended: true }));
app.use(bodyParser.urlencoded({ extended: true }));
app.use(cors());
app.use(express.json());
app.route('/').get((req, res) => {
    res.send('Hello World');
});

app.route('/register').post((req, res) => { 
    const { name, email, password } = req.body;
    const newUser = new User({
        name,
        email,
        password
    });
    newUser.save();
    res.send('User Registered');
});

app.route('/login').post((req, res) => {    
    const { email, password } = req.body;
    const user = User.findOne({ email: email, password: password })
    if (user) {
        id = user._id;
        res.send('Login Successful');
        
    } else {
        res.send('Login Failed');
    }
    }
    );



app.listen(3001, () => {  
    console.log('Server is running on port 3001');
});