/*
This is a game demo written by Sai-Keung Wong.
OGRE is employed for graphics rendering.
Date: Dec 2006 - Nov. 2020
Email: wingo.wong@gmail.com

*/
#include "wago_ogre_mesh.h"
//#include "common.h"
using namespace std;

GRAY_SCALE_IMAGE convertImageToGrayScaleImage(const Image& image)
{
	int width = image.getWidth();
	int height = image.getHeight();
	std::vector<float> image_array (width*height);

	// compute the number of bytes per pixel (byte per pixel)
	int bpp = int(image.getSize() / (width*height)); 
	
	int offset = 0;
	if (bpp == 4) {
		offset = 1;
	}

	//uint maxValue = (1 << (bpp*8)) - 1;
	uint maxValue = ((1 << 8) - 1)*(bpp-offset);


	const uchar* imageData = image.getData();
	
	/*
	//treat the bpp number of bytes as an integer
	//in conversion.
	for (int i = 0; i < image_array.size(); ++i)
	{
		uint val = 0; // should be at least bpp number of bytes in val
		memcpy(&val, imageData, bpp);
		imageData += bpp;
		
		image_array[i] = float(val) / maxValue;
	}
	*/
	
	for (int i = 0; i < image_array.size(); ++i)
	{
		uint tval = 0;
		uint val = 0; // should be at least bpp number of bytes in val
		for (int j = 0; j < bpp; j++) {
			memcpy(&val, imageData+j+offset, 1);
			tval += val;
		}
		imageData += bpp;
		
		image_array[i] = float(tval) / maxValue;
	}

	return GRAY_SCALE_IMAGE(image_array, width, height);
}


void convertImageToGrayScaleImage(const Image& image, float *image_array)
{
	int width = image.getWidth();
	int height = image.getHeight();

	// compute the number of bytes per pixel (byte per pixel)
	int bpp = int(image.getSize() / (width*height));


	uint maxValue = (1 << (bpp*8)) - 1;


	const uchar* imageData = image.getData();


	//treat the bpp number of bytes as an integer
	//in conversion.
	
	for (int i = 0; i < width*height; ++i)
	{
		uint val = 0; // should be at least bpp number of bytes in val
		memcpy(&val, imageData, bpp);
		imageData += bpp;
		image_array[i] = float(val) / maxValue;
	}
}

GRAY_SCALE_IMAGE::GRAY_SCALE_IMAGE()
: mWidth(0), mHeight(0), mArray(0)
{
}

GRAY_SCALE_IMAGE::GRAY_SCALE_IMAGE(const float* array, size_t width, size_t height)
: mWidth(width), mHeight(height)
{
	mArray = new float[width*height];
	memcpy(mArray, array, width*height*sizeof(float));
}


GRAY_SCALE_IMAGE::GRAY_SCALE_IMAGE(const std::vector<float>& array, size_t width, size_t height)
: mWidth(width), mHeight(height)
{
	mArray = new float[width*height];
	copy(array.begin(), array.end(), mArray);
}

GRAY_SCALE_IMAGE::GRAY_SCALE_IMAGE(const GRAY_SCALE_IMAGE& g)
: mWidth(g.mWidth), mHeight(g.mHeight)
{
	mArray = new float[mWidth*mHeight];
	memcpy(mArray, g.mArray, mWidth*mHeight*sizeof(float));
}

GRAY_SCALE_IMAGE::~GRAY_SCALE_IMAGE()
{
	delete[] mArray;
}

void GRAY_SCALE_IMAGE::set(const float* array, size_t width, size_t height)
{
	mWidth = width;
	mHeight = height;
	mArray = new float[width*height];
	memcpy(mArray, array, width*height*sizeof(float));
}

WAGO_OGRE_MESH_OBJ::WAGO_OGRE_MESH_OBJ(SceneManager *m)
{
	mImage = NULL;
	mSceneMgr = m;
	_vertexBuf = NULL;
	_normalBuf = NULL;
	_txtBuf = NULL;
	mFaceIndices = NULL;
	//
	_vertexBuf_old = NULL;
	mNode = NULL;
	mEntity = NULL;
	//
	mNumVertices = 0;
	mNumFaces = 0;
	mNumLines = 0;
	mMaterialName.clear();
}

void WAGO_OGRE_MESH_OBJ::allocateLargeMemory()
{
	int maxv = 1024*1024;
	_vertexBuf = new float[maxv*3];
	_normalBuf = new float[maxv*3];
	_txtBuf = new float[maxv*2];
	mFaceIndices = new unsigned short[maxv*3];
	//
	_vertexBuf_old = new float[maxv*3];
}

void WAGO_OGRE_MESH_OBJ::bgnFace()
{
	mNumFaces = 0;
}

void WAGO_OGRE_MESH_OBJ::endFace()
{
}


void WAGO_OGRE_MESH_OBJ::bgnVertex()
{
	mNumVertices = 0;
}

void WAGO_OGRE_MESH_OBJ::endVertex()
{
}

void WAGO_OGRE_MESH_OBJ::addVertex(float x, float y, float z, float txt_x, float txt_y, float nx, float ny, float nz)
{
	_vertexBuf[mNumVertices*3+0] = x;
	_vertexBuf[mNumVertices*3+1] = y;
	_vertexBuf[mNumVertices*3+2] = z;

	if (_vertexBuf_old) {
		_vertexBuf_old[mNumVertices*3+0] = x;
		_vertexBuf_old[mNumVertices*3+1] = y;
		_vertexBuf_old[mNumVertices*3+2] = z;
	}

	_txtBuf[mNumVertices*2+0] = txt_x;
	_txtBuf[mNumVertices*2+1] = txt_y;
	_normalBuf[mNumVertices*3+0] = nx;
	_normalBuf[mNumVertices*3+1] = ny;
	_normalBuf[mNumVertices*3+2] = nz;
	mNumVertices++;
}

void WAGO_OGRE_MESH_OBJ::addFace(unsigned short i0, unsigned short i1, unsigned short i2) 
{
	mFaceIndices[mNumFaces*3+0] = i0;
	mFaceIndices[mNumFaces*3+1] = i1;
	mFaceIndices[mNumFaces*3+2] = i2;
	mNumFaces++;
}

void WAGO_OGRE_MESH_OBJ::addLine(unsigned short i0, unsigned short i1)
{
	mFaceIndices[mNumLines*2+1] = i0;
	mFaceIndices[mNumLines*2+2] = i1;
	mNumLines++;
}

void WAGO_OGRE_MESH_OBJ::bgnLine()
{
	mNumLines = 0;
}

void WAGO_OGRE_MESH_OBJ::endLine()
{
}

void WAGO_OGRE_MESH_OBJ::buildMesh(HardwareVertexBufferSharedPtr posVertexBuffer)
{
	posVertexBuffer->writeData(0,
		posVertexBuffer->getSizeInBytes(), // size
		_vertexBuf, // source
		true); // discard?

}

const AxisAlignedBox &WAGO_OGRE_MESH_OBJ::computeAABB()
{

	float mx, my, mz;
	float Mx, My, Mz;
	Mx = mx = _vertexBuf[0];
	My = my = _vertexBuf[1];
	Mz = mz = _vertexBuf[2];
	for (int i = 1; i < mNumVertices; i++) {
		if (_vertexBuf[i*3+0] < mx) mx = _vertexBuf[i*3+0];
		else if (_vertexBuf[i*3+0] > Mx) Mx = _vertexBuf[i*3+0];
		if (_vertexBuf[i*3+1] < my) my = _vertexBuf[i*3+1];
		else if (_vertexBuf[i*3+1] > My) My = _vertexBuf[i*3+1];
		if (_vertexBuf[i*3+2] < mz) mz = _vertexBuf[i*3+2];
		else if (_vertexBuf[i*3+2] > Mz) Mz = _vertexBuf[i*3+2];
	}
	mAABB.setMinimum(mx, my, mz);
	mAABB.setMaximum(Mx, My, Mz);
	return mAABB;
}


void WAGO_OGRE_MESH_OBJ::createMesh(SceneNode *parent)
{
	String name;
	generateGlobalObjName(name);
	//
	mMesh = MeshManager::getSingleton().createManual(name,
		ResourceGroupManager::DEFAULT_RESOURCE_GROUP_NAME) ;
	SubMesh* subMesh = mMesh->createSubMesh();
	subMesh->useSharedVertices=false;

	int numVertices = mNumVertices;

	
		// static buffer for position and normals
		posVertexBuffer =
			HardwareBufferManager::getSingleton().createVertexBuffer(
			3*sizeof(float), // size of one vertex data
			numVertices, // number of vertices
			HardwareBuffer::HBU_DYNAMIC_WRITE_ONLY_DISCARDABLE, // usage
			//HardwareBuffer::HBU_STATIC_WRITE_ONLY,
			false); // no shadow buffer

		normalVertexBuffer =
			HardwareBufferManager::getSingleton().createVertexBuffer(
			3*sizeof(float), // size of one vertex data
			numVertices, // number of vertices
			//HardwareBuffer::HBU_DYNAMIC_WRITE_ONLY_DISCARDABLE, // usage
			HardwareBuffer::HBU_STATIC_WRITE_ONLY,
			false); // no shadow buffer

		
		normalVertexBuffer->writeData(0,
		normalVertexBuffer->getSizeInBytes(), // size
		_normalBuf, // source
		true); // discard?

		buildMesh(posVertexBuffer);


		texcoordsVertexBuffers =
			HardwareBufferManager::getSingleton().createVertexBuffer(
			2*sizeof(float), // size of one vertex data
			numVertices, // number of vertices
			HardwareBuffer::HBU_STATIC_WRITE_ONLY, // usage
			false); // no shadow buffer
		texcoordsVertexBuffers->writeData(0,
			texcoordsVertexBuffers->getSizeInBytes(), // size
			_txtBuf, // source
			true); // discard?

		indexBuffer =
			HardwareBufferManager::getSingleton().createIndexBuffer(
			HardwareIndexBuffer::IT_16BIT,
			mNumFaces*3,
			HardwareBuffer::HBU_STATIC_WRITE_ONLY);
		indexBuffer->writeData(0,
			indexBuffer->getSizeInBytes(),
			mFaceIndices,
			true); // true?
	

	// Initialize vertex data
	subMesh->vertexData = new VertexData();
	subMesh->vertexData->vertexStart = 0;
	subMesh->vertexData->vertexCount = numVertices;
	// first, set vertex buffer bindings
	VertexBufferBinding *vbind = subMesh->vertexData->vertexBufferBinding ;
	vbind->setBinding(0, posVertexBuffer);
	vbind->setBinding(1, normalVertexBuffer);
	vbind->setBinding(2, texcoordsVertexBuffers);
	// now, set vertex buffer declaration
	VertexDeclaration *vdecl = subMesh->vertexData->vertexDeclaration ;
	vdecl->addElement(0, 0, VET_FLOAT3, VES_POSITION);
	vdecl->addElement(1, 0, VET_FLOAT3, VES_NORMAL);
	vdecl->addElement(2, 0, VET_FLOAT2, VES_TEXTURE_COORDINATES);

	// Initialize index data
	subMesh->indexData->indexBuffer = indexBuffer;
	subMesh->indexData->indexStart = 0;
	subMesh->indexData->indexCount = mNumFaces*3;

	// set mesh bounds
	
	mAABB = computeAABB();
		
	mMesh->_setBounds(mAABB);
	mMesh->load();
	mMesh->touch();

	if (parent==NULL) {
		mNode = mSceneMgr->getRootSceneNode()->createChildSceneNode();
	} else {
		mNode = parent->createChildSceneNode();
	}

	mEntity = mSceneMgr->createEntity(name, name);
	if (!mMaterialName.empty()) {
		mEntity->setMaterialName(mMaterialName);
		mEntity->setCastShadows(false);
		//
		Ogre::MaterialPtr mp = MaterialManager::getSingleton().getByName(mMaterialName);
		//mp->setReceiveShadows(false);
	}
	//
	mEntity->setVisible(true);

	mNode->attachObject(mEntity);
}

void WAGO_OGRE_MESH_OBJ::setMaterial(const String &name)
{
	if (mEntity) {
		mEntity->setMaterialName(name);
	}
	mMaterialName = name;
}



void WAGO_OGRE_MESH_OBJ::scaleTextureCoords(float fx, float fy)
{
	for (int i = 0; i < mNumVertices; i++) {
		_txtBuf[i*2+0] *= fx;
		_txtBuf[i*2+1] *= fy;
	}
}

void WAGO_OGRE_MESH_OBJ::translateSceneNode(float x, float y, float z)
{
	mOriginX = x;
	mOriginZ = z;
	mNode->translate(x, y, z);
}

void WAGO_OGRE_MESH_OBJ::translate(float x, float y, float z)
{
	for (int i = 0; i < mNumVertices; i++) {
		_vertexBuf[i*3+0] += x;
		_vertexBuf[i*3+1] += y;
		_vertexBuf[i*3+2] += z;
	}
}

void WAGO_OGRE_MESH_OBJ::createTerrain()
{
	int nx, nz;
	nx = 51;
	nz = 51;
	float w, h;
	w = 4000;
	h = 4000;
	float dx, dz;
	dx = w/(nx-1);
	dz = h/(nz-1);
	mTerrainInfo = new TERRAIN_INFO(nx, nz, w, h, -w*0.5, -h*0.5, dx, dz);
	int i , j;
	bgnVertex();

	for (j = 0; j < nz; j++) {
		for (i = 0; i < nx ; i++) {
			float x, y, z;
			x = -w/2 + dx*i;
			y = (rand()%1000)/100.0 + 40;
			//y = 20;
			z = -w/2 + dz*j;
			mTerrainInfo->setHeightAtVertex(i, j, y);
			addVertex(x, y, z, x, z);
		}
	}
	endVertex();
	bgnFace();
	for (j = 0; j < nz-1; j++) {
		for (i = 0; i < nx-1 ; i++) {
			unsigned short i0, i1, i2, i3;
			i0 = j*(nx) + i;
			i1 = i0 + 1;
			i2 = i0 + (nx);
			i3 = i2+1;
			//addFace(i0, i1, i2);
			//addFace(i1, i3, i2);
			addFace(i0, i2, i1);
			addFace(i1, i2, i3);
		}
	}
	endFace();
	float txt_f_x = 0.01;
	float txt_f_y = 0.01;
	scaleTextureCoords(txt_f_x, txt_f_y);
	calculateVertexNormal_Terrain(nx, nz);
	createMesh();
}

void WAGO_OGRE_MESH_OBJ::calculateVertexNormal_Terrain(size_t nx, size_t nz)
{
	size_t k = 0;
	for (size_t j = 0; j < nz; j++) {
		for (size_t i = 0; i < nx ; i++) {
			Vector3 norm = mTerrainInfo->getNormalAtVertex((int)i, (int)j);

			_normalBuf[(int)k++] = norm.x;
			_normalBuf[(int)k++] = norm.y;
			_normalBuf[(int)k++] = norm.z;
		}
	}
}

void WAGO_OGRE_MESH_OBJ::calculateVertexNormal_Terrain(int sx, int sz, size_t nx, size_t nz)
{
	size_t k = 0;
	for (size_t j = 0; j < nz; j++) {
		for (size_t i = 0; i < nx ; i++) {
			Vector3 norm = mTerrainInfo->getNormalAtVertex((int)i+sx, (int)j+sz);

			_normalBuf[(int)k++] = norm.x;
			_normalBuf[(int)k++] = norm.y;
			_normalBuf[(int)k++] = norm.z;
		}
	}
}

void WAGO_OGRE_MESH_OBJ::animate_Terrain(Real time_step) 
{

	int nx = 51;
	int nz = 51;
	int k = 0;
	static Real a;
	a += time_step;
	for (int j = 0; j < nz; j++) {
		for (int i = 0; i < nx ; i++) {

			k++;
			_vertexBuf[k] = sin(a)*100 + _vertexBuf_old[k];
			k++;
			k++;
		}
	}

	buildMesh(posVertexBuffer);

	calculateVertexNormal_Terrain(51, 51);

	buildMesh(normalVertexBuffer);
}

void WAGO_OGRE_MESH_OBJ::createDftModel_Wall_Horizontal()
{
	float w2 = 2000;
	float h2 = 200;
	float d2 = 60;
	float w = w2/2;
	float h = h2/2;
	float d = d2/2;

	float txt_f_x = 0.0035;
	float txt_f_y = 0.01;
	bgnVertex();
	addVertex(-w, h, d, -w, h, 0, 0, 1);
	addVertex(-w, -h, d, -w, -h, 0, 0, 1);
	addVertex(w, -h, d, w, -h, 0, 0, 1);
	addVertex(w, h, d, w, h, 0, 0, 1);	
	//
	addVertex(w, h, d, d, h, 1, 0, 0);
	addVertex(w, -h, d, d, -h, 1, 0, 0);
	addVertex(w, -h, -d, -d, -h, 1, 0, 0);
	addVertex(w, h, -d, -d, h, 1, 0, 0);	
	//
	addVertex(w, h, -d, w, h, 0, 0, -1);
	addVertex(w, -h, -d, w, -h, 0, 0, -1);
	addVertex(-w, -h, -d, -w, -h, 0, 0, -1);
	addVertex(-w, h, -d, -w, h, 0, 0, -1);	
	//
	addVertex(-w, h, -d, -d, h, -1, 0, 0);
	addVertex(-w, -h, -d, -d, -h, -1, 0, 0);
	addVertex(-w, -h, d, d, -h, -1, 0, 0);
	addVertex(-w, h, d, d, h, -1, 0, 0);	
	//
	addVertex(-w, h, -d, -d, -w, 0, 1, 0);
	addVertex(-w, h, d, d, -w, 0, 1, 0);
	addVertex(w, h, d, d, w, 0, 1, 0);
	addVertex(w, h, -d, -d, w, 0, 1, 0);	
	//
	addVertex(-w, -h, -d, -d, -w, 0, -1, 0);
	addVertex(w, -h, -d, -d, w, 0, -1, 0);	
	addVertex(w, -h, d, d, w, 0, -1, 0);
	addVertex(-w, -h, d, d, -w, 0, -1, 0);


	//
	endVertex();
	bgnFace();
	for (int i = 0, j = 0; j < 6; j++, i+=4) {
		addFace(i+0, i+1, i+2);
		addFace(i+0, i+2, i+3);
	}

	endFace();
	scaleTextureCoords(txt_f_x, txt_f_y);
	translate(0, h, 0);
	createMesh();


}


void WAGO_OGRE_MESH_OBJ::createDftModel_Wall_Vertical()
{
	float w2 = 60;
	float h2 = 200;
	float d2 = 2000;
	float w = w2/2;
	float h = h2/2;
	float d = d2/2;

	float txt_f_x = 0.0035;
	float txt_f_y = 0.01;
	bgnVertex();
	addVertex(-w, h, d, -w, h, 0, 0, 1);
	addVertex(-w, -h, d, -w, -h, 0, 0, 1);
	addVertex(w, -h, d, w, -h, 0, 0, 1);
	addVertex(w, h, d, w, h, 0, 0, 1);	
	//
	addVertex(w, h, d, d, h, 1, 0, 0);
	addVertex(w, -h, d, d, -h, 1, 0, 0);
	addVertex(w, -h, -d, -d, -h, 1, 0, 0);
	addVertex(w, h, -d, -d, h, 1, 0, 0);	
	//
	addVertex(w, h, -d, w, h, 0, 0, -1);
	addVertex(w, -h, -d, w, -h, 0, 0, -1);
	addVertex(-w, -h, -d, -w, -h, 0, 0, -1);
	addVertex(-w, h, -d, -w, h, 0, 0, -1);	
	//
	addVertex(-w, h, -d, -d, h, -1, 0, 0);
	addVertex(-w, -h, -d, -d, -h, -1, 0, 0);
	addVertex(-w, -h, d, d, -h, -1, 0, 0);
	addVertex(-w, h, d, d, h, -1, 0, 0);	
	//
	addVertex(-w, h, -d, -d, -w, 0, 1, 0);
	addVertex(-w, h, d, d, -w, 0, 1, 0);
	addVertex(w, h, d, d, w, 0, 1, 0);
	addVertex(w, h, -d, -d, w, 0, 1, 0);	
	//
	addVertex(-w, -h, -d, -d, -w, 0, -1, 0);
	addVertex(w, -h, -d, -d, w, 0, -1, 0);	
	addVertex(w, -h, d, d, w, 0, -1, 0);
	addVertex(-w, -h, d, d, -w, 0, -1, 0);


	//
	endVertex();
	bgnFace();
	for (int i = 0, j = 0; j < 6; j++, i+=4) {
		addFace(i+0, i+1, i+2);
		addFace(i+0, i+2, i+3);
	}
	endFace();
	scaleTextureCoords(txt_f_x, txt_f_y);
	translate(0, h, 0);
	createMesh();


}




void WAGO_OGRE_MESH_OBJ::createWallsBasedOnBitmap( )
{
    createWallsBasedOnBitmap("wago_game_map01.png");
}

void WAGO_OGRE_MESH_OBJ::createWallsBasedOnBitmap(
    const String &imageName
    )
{

	if (mImage==NULL) mImage = new Image;
	
	mImage->load(imageName, "General");
	
	//silly method to creat gray image but it works.
	GRAY_SCALE_IMAGE c = convertImageToGrayScaleImage(*mImage);
	mGrayImage.set(c.getImage(), c.getWidth(), c.getHeight());
	mMapImage.set(c.getImage(), c.getWidth(), c.getHeight());
	GRAY_SCALE_IMAGE *b = &mGrayImage;
	
	float dx = 0, dz = 0;
	mDX = dx = 10;
	mDZ = dz = 10;
	float h = 800;
	int nx = b->getWidth();
	int nz = b->getHeight();
	int i, j;
	float x, z;
	bool flg_start;
	Vector3 v0, v1, v2, v3;
	int nf = 0;
	bgnVertex();

	// bottom
	for (j = 0; j < nz; j++) {
		z = j*dz;
		flg_start = false;
		for (i = 0; i < nx; i++) {
			x = i*dx;
			if ((b->at(i, j) > 0.5) && !b->gr(i, j+1, 0.5)) {
				if(flg_start) {
				    if (b->gr(i+1, j, 0.5)) continue;
					    v0 = Vector3(x+dx, 0, z+dz);
					    v3 = Vector3(x+dx, h, z+dz);
					    Vector3 normal = Vector3(0, 0, -1);
					    addVertex(v1.x, v1.y, v1.z, v1.x, 0, -1, 0, 0);
					    addVertex(v0.x, v0.y, v0.z, v0.x, 0, -1, 0, 0);
					    addVertex(v3.x, v3.y, v3.z, v3.x, h, -1, 0, 0);
					    addVertex(v2.x, v2.y, v2.z, v2.x, h, -1, 0, 0);
			            flg_start = false;
				        nf++;
				} else {
					v1 = Vector3(x, 0, z+dz);
					v2 = Vector3(x, h, z+dz);
					flg_start = true;
				}
			  } else {
				  if (flg_start) {
				     v0 = Vector3(x+dx, 0, z+dz);
				     v3 = Vector3(x+dx, h, z+dz);
					 addVertex(v1.x, v1.y, v1.z, v1.x, 0, -1, 0, 0);
					 addVertex(v0.x, v0.y, v0.z, v0.x, 0, -1, 0, 0);
					 addVertex(v3.x, v3.y, v3.z, v3.x, h, -1, 0, 0);
					 addVertex(v2.x, v2.y, v2.z, v2.x, h, -1, 0, 0);
			         flg_start = false;
				     nf++;
				  }
			}
		}
	}

	
	//top
	for (j = 0; j < nz; j++) {
		z = j*dz;
		flg_start = false;
		for (i = 0; i < nx; i++) {
			x = i*dx;
			if ((b->at(i, j) > 0.5) && !b->gr(i, j-1, 0.5)) {
				if (flg_start) {
					if (b->gr(i+1, j, 0.5)) continue;
					v0 = Vector3(x+dx, 0, z);
					v3 = Vector3(x+dx, h, z);
					addVertex(v0.x, v0.y, v0.z, v0.x, 0, 0, 0, -1);
					addVertex(v1.x, v1.y, v1.z, v1.x, 0, 0, 0, -1);
					addVertex(v2.x, v2.y, v2.z, v2.x, h, 0, 0, -1);
					addVertex(v3.x, v3.y, v3.z, v3.x, h, 0, 0, -1);
					flg_start = false;
					nf++;
				} else {
					v1 = Vector3(x, 0, z);
					v2 = Vector3(x, h, z);
					flg_start = true;
				}
			} else {
				if (flg_start) {
					v0 = Vector3(x+dx, 0, z);
					v3 = Vector3(x+dx, h, z);
					addVertex(v0.x, v0.y, v0.z, v0.x, 0, 0, 0, -1);
					addVertex(v1.x, v1.y, v1.z, v1.x, 0, 0, 0, -1);
					addVertex(v2.x, v2.y, v2.z, v2.x, h, 0, 0, -1);
					addVertex(v3.x, v3.y, v3.z, v3.x, h, 0, 0, -1);
					flg_start = false;
					nf++;
				}
			}
		}
	}
    
	
	//left
	for (i = 0; i < nx; i++) {
		x = i*dx;
		flg_start = false;

		for (j = 0; j < nz; j++) {
			z = j*dz;
			if ((b->at(i, j) > 0.5) && !b->gr(i-1, j, 0.5)) {
				if (flg_start) {
					if (b->gr(i, j+1, 0.5)) continue;
					v1 = Vector3(x, 0, z+dz);
					v2 = Vector3(x, h, z+dz);
					addVertex(v0.x, v0.y, v0.z, v0.z, 0, -1, 0, 0);
					addVertex(v1.x, v1.y, v1.z, v1.z, 0, -1, 0, 0);
					addVertex(v2.x, v2.y, v2.z, v2.z, h, -1, 0, 0);
					addVertex(v3.x, v3.y, v3.z, v3.z, h, -1, 0, 0);
					flg_start = false;
					nf++;
				} else {
					v0 = Vector3(x, 0, z);
					v3 = Vector3(x, h, z);
					flg_start = true;
				}
			} else {
				if (flg_start) {
					//if (j>=nz-1) {
					v1 = Vector3(x, 0, z+dz);
					v2 = Vector3(x, h, z+dz);
					//} else {
					//	v1 = Vector3(x, 0, z);
					//	v2 = Vector3(x, h, z);
					//}
					addVertex(v0.x, v0.y, v0.z, v0.z, 0, -1, 0, 0);
					addVertex(v1.x, v1.y, v1.z, v1.z, 0, -1, 0, 0);
					addVertex(v2.x, v2.y, v2.z, v2.z, h, -1, 0, 0);
					addVertex(v3.x, v3.y, v3.z, v3.z, h, -1, 0, 0);
					flg_start = false;
					nf++;
				}
			}
		}
	}
    
	
	//right
	for (i = 0; i < nx; i++) {
		x = i*dx;
		flg_start = false;

		for (j = 0; j < nz; j++) {
			z = j*dz;
			if ((b->at(i, j) > 0.5) && !b->gr(i+1, j, 0.5)) {
				//cout << "..." << endl;
				if (flg_start) {
					if (b->gr(i, j+1, 0.5)) continue;
					v0 = Vector3(x+dx, 0, z);
				    v3 = Vector3(x+dx, h, z);
					Vector3 normal = Vector3(0, 0, -1);
					addVertex(v0.x, v0.y, v0.z, v0.z, 0, normal.x, normal.y, normal.z);
					addVertex(v1.x, v1.y, v1.z, v1.z, 0, normal.x, normal.y, normal.z);
					addVertex(v2.x, v2.y, v2.z, v2.z, h, normal.x, normal.y, normal.z);
					addVertex(v3.x, v3.y, v3.z, v3.z, h, normal.x, normal.y, normal.z);
					flg_start = false;
					nf++;
				} else {
					v1 = Vector3(x+dx, 0, z);
					v2 = Vector3(x+dx, h, z);
					flg_start = true;
				}
			} else {
				if (flg_start) {
					v0 = Vector3(x+dx, 0, z);
					v3 = Vector3(x+dx, h, z);
					Vector3 normal = Vector3(0, 0, -1);
					addVertex(v0.x, v0.y, v0.z, v0.z, 0, normal.x, normal.y, normal.z);
					addVertex(v1.x, v1.y, v1.z, v1.z, 0, normal.x, normal.y, normal.z);
					addVertex(v2.x, v2.y, v2.z, v2.z, h, normal.x, normal.y, normal.z);
					addVertex(v3.x, v3.y, v3.z, v3.z, h, normal.x, normal.y, normal.z);
					flg_start = false;
					nf++;
				}
			}
		}
	}
	//
	for (j = 0; j < nz; j++) {
		z = j*dz;
		flg_start = false;
		int k = 0;
		for (i = 0; i < nx; i++) {
			x = i*dx;

			if (b->at(i, j) > 0.5) {
				k++;
				if (flg_start) {
				} else {
					v3 = Vector3(x, h, z);
					v0 = Vector3(x, h, z+dz);
					flg_start = true;
				}
			} else {
				if (k>0) {
					v2 = Vector3(v3.x+dx*k, h, z);
					v1 = Vector3(v3.x+dx*k, h, z+dz);
					addVertex(v0.x, v0.y, v0.z, v0.x, v0.z, 0, 1, 0);
					addVertex(v1.x, v1.y, v1.z, v1.x, v1.z, 0, 1, 0);
					addVertex(v2.x, v2.y, v2.z, v2.x, v2.z, 0, 1, 0);
					addVertex(v3.x, v3.y, v3.z, v3.x, v3.z, 0, 1, 0);
					nf++;
					k = 0;
					flg_start = false;
				}
			}
		}
		if (k>0) {
			v2 = Vector3(v3.x+dx*k, h, z);
			v1 = Vector3(v3.x+dx*k, h, z+dz);
			addVertex(v0.x, v0.y, v0.z, v0.x, v0.z, 0, 1, 0);
			addVertex(v1.x, v1.y, v1.z, v1.x, v1.z, 0, 1, 0);
			addVertex(v2.x, v2.y, v2.z, v2.x, v2.z, 0, 1, 0);
			addVertex(v3.x, v3.y, v3.z, v3.x, v3.z, 0, 1, 0);
			nf++;
			k = 0;
			flg_start = false;
		}
	}
	//
	endVertex();
	bgnFace();
	int k = 0;
	for (i = 0, k = 0; i < nf; i++, k+=4) {
		unsigned short i0, i1, i2, i3;
		i0 = k;
		i1 = k+1;
		i2 = k+2;
		i3 = k+3;
		addFace(i0, i1, i2);
		addFace(i0, i2, i3);
	}
	endFace();
	float txt_f_x = 0.0035;
	float txt_f_y = 0.01;
	scaleTextureCoords(txt_f_x, txt_f_y);
	createMesh();
}

SceneNode* WAGO_OGRE_MESH_OBJ::createTerrainFromGrayLevelImage_Portion(SceneNode *parent, size_t sx, size_t sz, size_t nx, size_t nz, float dx, float dz, float size_x, float size_z)
{

	size_t i, j;
	float x, y, z;

	bgnVertex();

	for (j = 0; j < nz; j++) {
		z = j*dz + sz*dz;
		for (i = 0; i < nx; i++) {
			x = i*dx + sx*dx;
			y = mTerrainInfo->getHeightAtVertex(i+sx, j+sz);
			addVertex(x, y, z, x, z);
		}
	}
	endVertex();
	
	bgnFace();
	for (j = 0; j < nz-1; j++) {
		for (i = 0; i < nx-1; i++) {
			unsigned short i0, i1, i2, i3;
			i0 = j*nx+i;
			i1 = i0+1;
			i2 = i1 + nx;
			i3 = i0 + nx;
			addFace(i0, i2, i1);
			addFace(i0, i3, i2);
		}
	}
	endFace();

	float txt_f_x = 1/(size_x);
	float txt_f_z = 1/(size_z);
	scaleTextureCoords(txt_f_x, txt_f_z);
	translate(-size_x/2.0, 0, -size_z/2.0);
	calculateVertexNormal_Terrain(nx, nz);

	createMesh(parent);

	return mNode;
}

SceneNode *WAGO_OGRE_MESH_OBJ::createTerrainFromGrayLevelImage(const GRAY_SCALE_IMAGE &b, float dx, float dz, float h)
{
	SceneNode *parent_node = mSceneMgr->getRootSceneNode()->createChildSceneNode();

	size_t nx = b.getWidth();
	size_t nz = b.getHeight();

	size_t i, j;

	float x, y, z;
	float size_x = (nx-1)*dx;
	float size_z = (nz-1)*dz;

	mTerrainInfo = new TERRAIN_INFO(nx, nz, size_x, size_z, -size_x/2.0, -size_z/2.0, dx, dz);
	for (j = 0; j < nz; j++) {
		z = j*dz;
		for (i = 0; i < nx; i++) {
			x = i*dx;
			y = b.at(i, j)*h;
			mTerrainInfo->setHeightAtVertex(i, j, y);		
		}
	}

	
	int ndx = 32;
	int ndz = 32;

	int m_ndx;
	int m_ndz;
	int sz = 0;
	int sx;



	bool flg = true;
	while (sz < nz && flg) {
		m_ndz = ndz;
		sx = 0;
		if (sz + m_ndz > nz) {
			m_ndz = nz - sz;
		}
		while (sx < nx && flg) {
			m_ndx = ndx;
			if (sx + m_ndx > nx) {
				m_ndx = nx - sx;
			}
			
			createTerrainFromGrayLevelImage_Portion(
				parent_node,
				sx, sz,
				m_ndx, m_ndz,
				dx, dz,
				size_x, size_z);
			
			sx += ndx-1;
			
		}
		sz += ndz-1;
		
	}

	return parent_node;
}

SceneNode *WAGO_OGRE_MESH_OBJ::createTerrainFromGrayLevelImage_Entire(const GRAY_SCALE_IMAGE &b, float dx, float dz, float h)
{
	size_t nx = b.getWidth();
	size_t nz = b.getHeight();
	size_t i, j;
	float x, y, z;

	float size_x = dx*(nx-1);
	float size_z = dz*(nz-1);
	mTerrainInfo = new TERRAIN_INFO(nx, nz, size_x, size_z, -size_x*0.5, -size_z*0.5, dx, dz);
	bgnVertex();

	for (j = 0; j < nz; j++) {
		z = j*dz;
		for (i = 0; i < nx; i++) {
			x = i*dx;
			y = b.at(i, j)*h;
			mTerrainInfo->setHeightAtVertex(i, j, y);
			addVertex(x, y, z, x, z);
		}
	}
	endVertex();
	bgnFace();
	for (j = 0; j < nz-1; j++) {
		for (i = 0; i < nx-1; i++) {
			unsigned short i0, i1, i2, i3;
			i0 = (unsigned short)(j*nx+i);
			i1 = (unsigned short)(i0+1);
			i2 = (unsigned short)(i1 + nx);
			i3 = (unsigned short)(i0 + nx);
			addFace(i0, i2, i1);
			addFace(i0, i3, i2);
		}
	}
	endFace();

	float txt_f_x = 1/(dx*(nx-1));
	float txt_f_z = 1/(dz*(nz-1));
	scaleTextureCoords(txt_f_x, txt_f_z);
	translate(-dx*(nx-1)/2.0, 0, -dz*(nz-1)/2.0);
	calculateVertexNormal_Terrain(nx, nz);

	createMesh();
	return mNode;
}

void WAGO_OGRE_MESH_OBJ::create(const String &imageName)
{
	//return;
	allocateLargeMemory();
	//createDftModel_Wall_Horizontal();
	//createDftModel_Wall_Vertical();
	//createTerrain();
	createWallsBasedOnBitmap(imageName);
	//Ogre::MeshSerializer serializer;
	//serializer.exportMesh(mMesh.getPointer(), "wao_map01.mesh");
}

void WAGO_OGRE_MESH_OBJ::create()
{
	//return;
	allocateLargeMemory();
	//createDftModel_Wall_Horizontal();
	//createDftModel_Wall_Vertical();
	//createTerrain();
	createWallsBasedOnBitmap();
	//Ogre::MeshSerializer serializer;
	//serializer.exportMesh(mMesh.getPointer(), "wao_map01.mesh");
}

void WAGO_OGRE_MESH_OBJ::animate(Real time_step) 
{
	//animate_Terrain(time_step);
}

WAGO_OGRE_MESH_OBJ::~WAGO_OGRE_MESH_OBJ()
{
	//cout << "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!delete!!!!!!!!!!!!" << endl;
	if (mFaceIndices) delete [] mFaceIndices;
	if (_vertexBuf_old) delete [] _vertexBuf_old;
	if (_vertexBuf) delete [] _vertexBuf;
	if (_normalBuf) delete [] _normalBuf;
	if (_txtBuf) delete [] _txtBuf;

	if (mNode) {
		mNode->detachAllObjects();
		mSceneMgr->destroySceneNode(mNode->getName());
		mSceneMgr->destroyEntity(mEntity);
	}
	mMaterialName.clear();
} 

//-------------------------------------------------------------
SIMPLE_TERRAIN::SIMPLE_TERRAIN(SceneManager *m) : WAGO_OGRE_MESH_OBJ(m)
{
        mNormalVecMap = 0;
    mNVM_Width = 0;
    mNVM_Height = 0;
}

void SIMPLE_TERRAIN::dilateMapObstacles(int a_Amount)
{
	mMapImage.dilate(a_Amount, &mGrayImage);
}

void SIMPLE_TERRAIN::getDestination(Vector3 &d)
{
	d = mDestination;
}

void SIMPLE_TERRAIN::getStartingPosition(Vector3 &d)
{
	d = mStartingPosition;
}

float SIMPLE_TERRAIN::getGridCellValue(float x, float z)
{
	int xi = (x - mOriginX)/mDX;
	int zi = (z - mOriginZ)/mDZ;
	return mMapImage.atWithBoundCheck(xi, zi);
	 
}

void SIMPLE_TERRAIN::scanMapForLocatingObjects()
{
	int nx = mImage->getWidth();
	int nz = mImage->getHeight();
	int bpp = int(mImage->getSize() / (nx*nz)); 
	if (bpp < 3) return;
	int offset = 0;
	if (bpp == 4) offset = 1;
	const uchar* imageData = mImage->getData();
	
	uint t = 100; //threshold
	for (int j = 0; j < nz; ++j)
	{
		for (int i = 0; i < nx; ++i) {
			uint index = (i + j*nx)*bpp;
			
			uint b = imageData[index+0+offset]; //blue color
			uint g = imageData[index+1+offset]; //green color
			uint r = imageData[index+2+offset]; //red color
			if (r > t && g > t && b > t) continue;
			if (r < t && g < t && b < t) continue;
			float x, z;
			x = i*mDX+mOriginX;
			z = j*mDZ+mOriginZ;
			//DEBUG_LOG_MSG_3INT("color R", i, j, r);
			//DEBUG_LOG_MSG_3INT("color G", i, j, g);
			//DEBUG_LOG_MSG_3INT("color B", i, j, b);
			if (b > t) {
				//blue
				mStartingPosition = Vector3(x, 1000, z);
			}

			if (r > t) {
				//red
				mMonsterLocation.push_back(Vector3(x, 1000, z));
			}
			
			if (g > t) { 
				//green
				mDestination = Vector3(x, 0, z);
			}
		}
	}
}

const std::vector<Vector3> &SIMPLE_TERRAIN::getLocationOfMonsters() const
{
	return mMonsterLocation;
}

void SIMPLE_TERRAIN::computeNormalVectors( )
{
    if (mNormalVecMap) return;
    mNVM_Width = mImage->getWidth();
	mNVM_Height = mImage->getHeight();

    mNormalVecMap = new Vector3[mNVM_Width*mNVM_Height];             // NVM 

    for (int i = 0; i <mNVM_Width*mNVM_Height; ++i )
    {
        mNormalVecMap[i] = Vector3(0.0, 0.0, 0.0);
    }
    Vector3 vec_top = Vector3( 0.0, 0.0, 1.0);
    Vector3 vec_bottom = Vector3( 0.0, 0.0, -1.0);
    Vector3 vec_right = Vector3( 1.0, 0.0, 0.0);
    Vector3 vec_left = Vector3( -1.0, 0.0, 0.0);

    int numDirections = 8;
    Vector3 dir[] = {
        Vector3(1,0, 0)
        , Vector3(0, 0, 1)
        , Vector3(-1, 0,0)
        , Vector3(0,0, -1)
        , Vector3(1,0, 1)
        , Vector3(-1,0, 1)
        , Vector3(1,0, -1)
        , Vector3(-1,0, -1)
    };
    for (int j = 0; j < numDirections; ++j) {
        dir[j].normalise( );
    }

    for (int y = 0; y < mNVM_Height; ++y) {

        for (int x = 0; x < mNVM_Width; ++x) {
            Vector3 n = Vector3(0.0, 0.0, 0.0);

            float v = mMapImage.atWithBoundCheck(x, y);
            /*
            float v_top = mMapImage.at_top(x, y);
            float v_left = mMapImage.at_left(x, y);
            float v_right = mMapImage.at_right(x, y);
            float v_bottom = mMapImage.at_bottom(x, y);
            if (v < v_top ) n += vec_bottom ;
            if (v < v_bottom ) n += vec_top;
            if (v < v_left ) n += vec_right;
            if (v < v_right ) n += vec_left;
            */
            for (int j = 0; j < numDirections; ++j) {
                int ix = dir[j].x;
                int iy = dir[j].z;
                ix += x;
                iy += y;
                float tmp_v = mMapImage.atWithBoundCheck(ix, iy);
                if ( v > tmp_v) {
                    n += dir[j];
                } else if (v < tmp_v) {
                    n -= dir[j];
                }
            }


            n.normalise( );

            int index = x + y*mNVM_Width;
            mNormalVecMap[index] = n;
        }
    }
}

Vector3 SIMPLE_TERRAIN::getGridNormalVector( float x, float z ) const
{
    Vector3 n;
    //cout << "mNormalVecMap:" << mNormalVecMap << endl;
 
     //   cout << "mNVM_Width:" <<mNVM_Width << endl;
    //cout << "mNVM_Height:" <<mNVM_Height << endl;

    //cout << "x:" <<x << endl;
    //cout << "z:" <<z << endl;
    int xi = (x - mOriginX)/mDX;
	int zi = (z - mOriginZ)/mDZ;
    if (xi < 0) xi = 0;
		else if (xi >= mNVM_Width) xi = mNVM_Width - 1;
		if (zi < 0) zi = 0;
		else if (zi >= mNVM_Height) zi = mNVM_Height - 1;
        //
    //        cout << "xi:" <<xi << endl;
    //cout << "zi:" <<zi << endl;
        //
		n = mNormalVecMap[xi + zi*mNVM_Width];
    //    cout << "normal:" << n.x << "\t" << n.y << "\t" << n.z << endl;
    
        return n;
}